from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


AGENT_NAMES = {
    "survey_agent",
    "innovation_agent",
    "method_agent",
    "limitation_agent",
}
MAX_SKILL_BYTES = 1024 * 1024
BUILTIN_SKILLS = {
    "survey_agent": ["survey.md"],
    "innovation_agent": ["innovation.md"],
    "method_agent": ["method.md"],
    "limitation_agent": ["limitation.md"],
}


@dataclass(frozen=True)
class AgentSkill:
    agent_name: str
    filename: str
    path: Path
    content: str
    source: str = "external"


class AgentSkillLibrary:
    def __init__(
        self,
        root: str | Path = "agent_skills",
        builtin_root: str | Path = "profiles",
    ):
        self.root = Path(root)
        self.builtin_root = Path(builtin_root)

    def agent_dir(self, agent_name: str) -> Path:
        if agent_name not in AGENT_NAMES:
            raise ValueError(f"Unknown agent: {agent_name}")
        return self.root / agent_name

    @staticmethod
    def safe_filename(filename: str) -> str:
        source_name = Path(filename).name
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(source_name).stem)
        stem = stem.strip(".-_") or "skill"
        return f"{stem}.md"

    def save(
        self,
        agent_name: str,
        filename: str,
        content: bytes,
        *,
        overwrite: bool = False,
    ) -> AgentSkill:
        if len(content) > MAX_SKILL_BYTES:
            raise ValueError("Skill 文件不能超过 1 MB。")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Skill 文件必须使用 UTF-8 编码。") from exc
        if not text.strip():
            raise ValueError("Skill 文件不能为空。")

        directory = self.agent_dir(agent_name)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / self.safe_filename(filename)
        builtin_names = {
            skill.filename for skill in self.list_builtin(agent_name)
        }
        conflicts_with_builtin = path.name in builtin_names
        if (path.exists() or conflicts_with_builtin) and not overwrite:
            raise FileExistsError(f"{path.name} 已存在。")
        path.write_text(text, encoding="utf-8")
        return AgentSkill(agent_name, path.name, path, text)

    def list(self, agent_name: str | None = None) -> list[AgentSkill]:
        agent_names = [agent_name] if agent_name else sorted(AGENT_NAMES)
        skills: list[AgentSkill] = []
        for name in agent_names:
            directory = self.agent_dir(name)
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.md")):
                skills.append(
                    AgentSkill(
                        agent_name=name,
                        filename=path.name,
                        path=path,
                        content=path.read_text(encoding="utf-8"),
                    )
                )
        return skills

    def list_builtin(
        self,
        agent_name: str | None = None,
    ) -> list[AgentSkill]:
        agent_names = [agent_name] if agent_name else sorted(AGENT_NAMES)
        skills: list[AgentSkill] = []
        for name in agent_names:
            for filename in BUILTIN_SKILLS.get(name, []):
                path = self.builtin_root / filename
                if not path.exists():
                    continue
                skills.append(
                    AgentSkill(
                        agent_name=name,
                        filename=path.name,
                        path=path,
                        content=path.read_text(encoding="utf-8"),
                        source="builtin",
                    )
                )
        return skills

    def list_installed(
        self,
        agent_name: str | None = None,
    ) -> list[AgentSkill]:
        return self.list_builtin(agent_name) + self.list(agent_name)

    def list_effective(
        self,
        agent_name: str,
    ) -> list[AgentSkill]:
        external = self.list(agent_name)
        external_names = {skill.filename for skill in external}
        builtin = [
            skill
            for skill in self.list_builtin(agent_name)
            if skill.filename not in external_names
        ]
        return builtin + external

    def get_installed(
        self,
        agent_name: str,
        filename: str,
        source: str,
    ) -> AgentSkill | None:
        if source not in {"builtin", "external"}:
            return None
        skills = (
            self.list_builtin(agent_name)
            if source == "builtin"
            else self.list(agent_name)
        )
        return next(
            (
                skill
                for skill in skills
                if skill.filename == Path(filename).name
            ),
            None,
        )

    def update(
        self,
        agent_name: str,
        filename: str,
        source: str,
        content: str,
    ) -> AgentSkill:
        skill = self.get_installed(agent_name, filename, source)
        if skill is None:
            raise FileNotFoundError("Skill 不存在或已被删除。")
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_SKILL_BYTES:
            raise ValueError("Skill 文件不能超过 1 MB。")
        if not content.strip():
            raise ValueError("Skill 文件不能为空。")

        skill.path.write_text(content, encoding="utf-8")
        return AgentSkill(
            agent_name=skill.agent_name,
            filename=skill.filename,
            path=skill.path,
            content=content,
            source=skill.source,
        )

    def combined_prompt(self, agent_name: str) -> str:
        sections = [
            f"## Skill: {skill.filename}\n{skill.content.strip()}"
            for skill in self.list_effective(agent_name)
        ]
        return "\n\n".join(sections)

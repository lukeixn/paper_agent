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


@dataclass(frozen=True)
class AgentSkill:
    agent_name: str
    filename: str
    path: Path
    content: str


class AgentSkillLibrary:
    def __init__(self, root: str | Path = "agent_skills"):
        self.root = Path(root)

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
        if path.exists() and not overwrite:
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

    def combined_prompt(self, agent_name: str) -> str:
        sections = [
            f"## Skill: {skill.filename}\n{skill.content.strip()}"
            for skill in self.list(agent_name)
        ]
        return "\n\n".join(sections)

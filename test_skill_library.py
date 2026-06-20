from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import agent.agent as agent_module
from agent.agent import SurveyAgent
from skill_library import AgentSkillLibrary


def test_skills_are_isolated_by_agent() -> None:
    with TemporaryDirectory() as temporary_directory:
        library = AgentSkillLibrary(temporary_directory)
        library.save(
            "survey_agent",
            "literature review.md",
            b"# Review Skill\nFocus on chronology.",
        )
        library.save(
            "method_agent",
            "methods.md",
            b"# Method Skill\nCompare implementation details.",
        )

        survey_prompt = library.combined_prompt("survey_agent")
        method_prompt = library.combined_prompt("method_agent")

        assert "Focus on chronology" in survey_prompt
        assert "Compare implementation details" not in survey_prompt
        assert "Compare implementation details" in method_prompt
        assert "Focus on chronology" not in method_prompt


def test_skill_filename_is_sanitized() -> None:
    with TemporaryDirectory() as temporary_directory:
        library = AgentSkillLibrary(temporary_directory)
        skill = library.save(
            "innovation_agent",
            "../../My Skill 中文.md",
            "# Innovation\nFind novelty.".encode("utf-8"),
        )

        assert skill.filename == "My-Skill.md"
        assert skill.path.parent == (
            Path(temporary_directory) / "innovation_agent"
        )


def test_agent_loads_its_external_skills() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        library = AgentSkillLibrary(root / "agent_skills")
        library.save(
            "survey_agent",
            "chronology.md",
            b"# Chronology\nAlways compare publication years.",
        )

        original_root = agent_module.PROJECT_ROOT
        try:
            agent_module.PROJECT_ROOT = root
            agent = SurveyAgent()
        finally:
            agent_module.PROJECT_ROOT = original_root

        assert "External Agent Skills" in agent.profile
        assert "Always compare publication years" in agent.profile


if __name__ == "__main__":
    test_skills_are_isolated_by_agent()
    test_skill_filename_is_sanitized()
    test_agent_loads_its_external_skills()
    print("skill library tests passed")

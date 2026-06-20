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

        assert "Agent Skills" in agent.profile
        assert "Always compare publication years" in agent.profile


def test_builtin_innovation_skill_is_listed() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        profiles = root / "profiles"
        profiles.mkdir()
        (profiles / "innovation.md").write_text(
            "# Built-in Innovation\nReview novelty.",
            encoding="utf-8",
        )
        library = AgentSkillLibrary(
            root / "agent_skills",
            profiles,
        )

        skills = library.list_installed("innovation_agent")

        assert len(skills) == 1
        assert skills[0].filename == "innovation.md"
        assert skills[0].source == "builtin"
        assert "Review novelty" in skills[0].content
        assert "Review novelty" in library.combined_prompt(
            "innovation_agent"
        )


def test_external_skill_can_override_builtin_skill() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        profiles = root / "profiles"
        profiles.mkdir()
        (profiles / "innovation.md").write_text(
            "# Built-in\nUse the original instruction.",
            encoding="utf-8",
        )
        library = AgentSkillLibrary(
            root / "agent_skills",
            profiles,
        )

        try:
            library.save(
                "innovation_agent",
                "innovation.md",
                b"# Override\nUse the replacement instruction.",
            )
            raise AssertionError("Expected overwrite confirmation")
        except FileExistsError:
            pass

        library.save(
            "innovation_agent",
            "innovation.md",
            b"# Override\nUse the replacement instruction.",
            overwrite=True,
        )
        prompt = library.combined_prompt("innovation_agent")
        effective = library.list_effective("innovation_agent")

        assert "Use the replacement instruction" in prompt
        assert "Use the original instruction" not in prompt
        assert len(effective) == 1
        assert effective[0].source == "external"


def test_all_agents_have_builtin_skills() -> None:
    library = AgentSkillLibrary()
    skills = library.list_builtin()

    assert {
        (skill.agent_name, skill.filename)
        for skill in skills
    } == {
        ("survey_agent", "survey.md"),
        ("innovation_agent", "innovation.md"),
        ("method_agent", "method.md"),
        ("limitation_agent", "limitation.md"),
    }
    for skill in skills:
        assert len(skill.content) > 300
        assert skill.source == "builtin"
        assert library.get_installed(
            skill.agent_name,
            skill.filename,
            "builtin",
        ) == skill
    assert library.get_installed(
        "innovation_agent",
        "../innovation.md",
        "builtin",
    ) is not None
    assert library.get_installed(
        "innovation_agent",
        "missing.md",
        "builtin",
    ) is None


def test_builtin_and_external_skills_can_be_updated() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        profiles = root / "profiles"
        profiles.mkdir()
        (profiles / "survey.md").write_text(
            "# Original Survey",
            encoding="utf-8",
        )
        library = AgentSkillLibrary(
            root / "agent_skills",
            profiles,
        )
        library.save(
            "method_agent",
            "custom.md",
            b"# Original Method",
        )

        builtin = library.update(
            "survey_agent",
            "survey.md",
            "builtin",
            "# Updated Survey\nUse evidence clusters.",
        )
        external = library.update(
            "method_agent",
            "custom.md",
            "external",
            "# Updated Method\nCompare reproducibility.",
        )

        assert builtin.source == "builtin"
        assert external.source == "external"
        assert "Use evidence clusters" in (
            profiles / "survey.md"
        ).read_text(encoding="utf-8")
        assert "Compare reproducibility" in library.combined_prompt(
            "method_agent"
        )


if __name__ == "__main__":
    test_skills_are_isolated_by_agent()
    test_skill_filename_is_sanitized()
    test_agent_loads_its_external_skills()
    test_builtin_innovation_skill_is_listed()
    test_external_skill_can_override_builtin_skill()
    test_all_agents_have_builtin_skills()
    test_builtin_and_external_skills_can_be_updated()
    print("skill library tests passed")

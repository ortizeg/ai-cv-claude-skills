"""Test that all agents are complete."""

from __future__ import annotations

from pathlib import Path

import pytest

AGENTS_DIR = Path("agents")


def get_all_agents() -> list[Path]:
    """Get all agent directories."""
    return sorted([d for d in AGENTS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])


@pytest.mark.parametrize("agent_dir", get_all_agents(), ids=lambda d: d.name)
def test_agent_has_skill_md(agent_dir: Path) -> None:
    """Test agent has SKILL.md."""
    skill_md = agent_dir / "SKILL.md"
    assert skill_md.exists(), f"Missing SKILL.md in {agent_dir.name}"
    assert skill_md.stat().st_size > 500, f"SKILL.md too short in {agent_dir.name}"


@pytest.mark.parametrize("agent_dir", get_all_agents(), ids=lambda d: d.name)
def test_agent_has_readme(agent_dir: Path) -> None:
    """Test agent has README."""
    readme = agent_dir / "README.md"
    assert readme.exists(), f"Missing README.md in {agent_dir.name}"


def test_blocking_agents_have_actions() -> None:
    """Test that blocking agents have GitHub Actions."""
    blocking_agents = ["code-review", "test-engineer"]

    for agent_name in blocking_agents:
        agent_dir = AGENTS_DIR / agent_name
        action_yml = agent_dir / "action.yml"
        assert action_yml.exists(), f"Missing action.yml for blocking agent {agent_name}"


def test_all_expected_agents_exist() -> None:
    """Test all expected agents exist."""
    expected = {"expert-coder", "ml-engineer", "code-review", "test-engineer"}
    actual = {d.name for d in get_all_agents()}

    missing = expected - actual
    assert not missing, f"Missing agents: {missing}"

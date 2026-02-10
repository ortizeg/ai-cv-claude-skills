"""Test that all agents are complete."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

AGENTS_DIR = Path("agents")

if sys.version_info >= (3, 12):
    import tomllib
else:
    import tomli as tomllib


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


@pytest.mark.parametrize("agent_dir", get_all_agents(), ids=lambda d: d.name)
def test_agent_has_toml(agent_dir: Path) -> None:
    """Test agent has agent.toml metadata file."""
    toml_path = agent_dir / "agent.toml"
    assert toml_path.exists(), f"Missing agent.toml in {agent_dir.name}"


def test_blocking_agents_have_actions() -> None:
    """Test that blocking agents have GitHub Actions."""
    for agent_dir in get_all_agents():
        toml_path = agent_dir / "agent.toml"
        if toml_path.exists():
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
            if data.get("agent", {}).get("type") == "blocking":
                action_yml = agent_dir / "action.yml"
                assert action_yml.exists(), (
                    f"Missing action.yml for blocking agent {agent_dir.name}"
                )


def test_minimum_agent_count() -> None:
    """Test that we have at least the expected number of agents."""
    agents = get_all_agents()
    assert len(agents) >= 4, f"Expected at least 4 agents, found {len(agents)}"

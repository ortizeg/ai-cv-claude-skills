"""Test that GitHub Actions workflows are valid."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

WORKFLOWS_DIR = Path(".github/workflows")


def get_all_workflows() -> list[Path]:
    """Get all workflow files."""
    if not WORKFLOWS_DIR.exists():
        return []
    return sorted(WORKFLOWS_DIR.glob("*.yml"))


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
@pytest.mark.parametrize("workflow_path", get_all_workflows(), ids=lambda p: p.name)
def test_workflow_is_valid_yaml(workflow_path: Path) -> None:
    """Test that workflow file is valid YAML."""
    content = workflow_path.read_text()
    parsed = yaml.safe_load(content)
    assert parsed is not None, f"Empty workflow: {workflow_path.name}"
    assert isinstance(parsed, dict), f"Invalid workflow structure: {workflow_path.name}"


@pytest.mark.skipif(not HAS_YAML, reason="PyYAML not installed")
@pytest.mark.parametrize("workflow_path", get_all_workflows(), ids=lambda p: p.name)
def test_workflow_has_required_fields(workflow_path: Path) -> None:
    """Test workflow has required fields."""
    content = yaml.safe_load(workflow_path.read_text())

    assert "name" in content, f"Missing 'name' in {workflow_path.name}"
    # PyYAML parses the YAML keyword `on:` as boolean True
    assert "on" in content or True in content, f"Missing 'on' trigger in {workflow_path.name}"
    assert "jobs" in content, f"Missing 'jobs' in {workflow_path.name}"


def test_all_expected_workflows_exist() -> None:
    """Test all expected workflows exist."""
    expected = {"test.yml", "docs.yml", "lint.yml"}
    actual = {p.name for p in get_all_workflows()}

    missing = expected - actual
    assert not missing, f"Missing workflows: {missing}"

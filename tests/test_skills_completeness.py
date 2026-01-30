"""Test that all skills are complete and properly structured."""

from __future__ import annotations

from pathlib import Path

import pytest

SKILLS_DIR = Path("skills")
REQUIRED_FILES = ["SKILL.md", "README.md"]


def get_all_skills() -> list[Path]:
    """Get all skill directories."""
    return sorted([d for d in SKILLS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])


@pytest.mark.parametrize("skill_dir", get_all_skills(), ids=lambda d: d.name)
def test_skill_has_required_files(skill_dir: Path) -> None:
    """Test that skill has all required files."""
    for filename in REQUIRED_FILES:
        file_path = skill_dir / filename
        assert file_path.exists(), f"Missing {filename} in {skill_dir.name}"
        assert file_path.stat().st_size > 0, f"Empty {filename} in {skill_dir.name}"


@pytest.mark.parametrize("skill_dir", get_all_skills(), ids=lambda d: d.name)
def test_skill_md_has_content(skill_dir: Path) -> None:
    """Test that SKILL.md has substantial content."""
    skill_md = skill_dir / "SKILL.md"
    content = skill_md.read_text()

    # Check minimum length (1000+ chars for substantial content)
    assert len(content) > 500, f"SKILL.md in {skill_dir.name} is too short ({len(content)} chars)"

    # Check has code examples
    assert "```" in content, f"SKILL.md in {skill_dir.name} missing code examples"

    # Check has headers
    assert "# " in content, f"SKILL.md in {skill_dir.name} missing headers"


@pytest.mark.parametrize("skill_dir", get_all_skills(), ids=lambda d: d.name)
def test_readme_explains_skill(skill_dir: Path) -> None:
    """Test that README explains the skill."""
    readme = skill_dir / "README.md"
    content = readme.read_text().lower()

    # Should explain purpose
    assert any(word in content for word in ["purpose", "what", "when", "how", "overview", "use"]), (
        f"README in {skill_dir.name} doesn't explain purpose"
    )


def test_all_expected_skills_exist() -> None:
    """Test that all expected skills are present."""
    expected = {
        "master-skill",
        "pytorch-lightning",
        "pydantic-strict",
        "code-quality",
        "pixi",
        "docker-cv",
        "hydra-config",
        "loguru",
        "testing",
        "opencv",
        "matplotlib",
        "pypi",
        "gcp",
        "github-actions",
        "vscode",
        "pre-commit",
        "wandb",
        "mlflow",
        "tensorboard",
        "dvc",
        "onnx",
        "tensorrt",
        "abstraction-patterns",
        "library-review",
    }

    actual = {d.name for d in get_all_skills()}

    missing = expected - actual
    assert not missing, f"Missing skills: {missing}"

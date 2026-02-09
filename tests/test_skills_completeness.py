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

    # Check minimum length
    assert len(content) > 500, f"SKILL.md in {skill_dir.name} is too short ({len(content)} chars)"

    # Check has code examples
    assert "```" in content, f"SKILL.md in {skill_dir.name} missing code examples"

    # Check has headers
    assert "# " in content, f"SKILL.md in {skill_dir.name} missing headers"


@pytest.mark.parametrize("skill_dir", get_all_skills(), ids=lambda d: d.name)
def test_skill_md_has_frontmatter(skill_dir: Path) -> None:
    """Test that SKILL.md has YAML frontmatter with name and description."""
    skill_md = skill_dir / "SKILL.md"
    content = skill_md.read_text()

    assert content.startswith("---"), f"SKILL.md in {skill_dir.name} missing YAML frontmatter"

    # Find closing delimiter
    lines = content.split("\n")
    end_idx = -1
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = i
            break
    assert end_idx > 0, f"SKILL.md in {skill_dir.name} has unclosed frontmatter"

    frontmatter = "\n".join(lines[1:end_idx])
    assert "name:" in frontmatter, f"SKILL.md in {skill_dir.name} frontmatter missing 'name'"
    assert "description:" in frontmatter, (
        f"SKILL.md in {skill_dir.name} frontmatter missing 'description'"
    )


@pytest.mark.parametrize("skill_dir", get_all_skills(), ids=lambda d: d.name)
def test_skill_has_toml(skill_dir: Path) -> None:
    """Test that skill has a skill.toml metadata file."""
    toml_path = skill_dir / "skill.toml"
    assert toml_path.exists(), f"Missing skill.toml in {skill_dir.name}"


@pytest.mark.parametrize("skill_dir", get_all_skills(), ids=lambda d: d.name)
def test_readme_explains_skill(skill_dir: Path) -> None:
    """Test that README explains the skill."""
    readme = skill_dir / "README.md"
    content = readme.read_text().lower()

    assert any(word in content for word in ["purpose", "what", "when", "how", "overview", "use"]), (
        f"README in {skill_dir.name} doesn't explain purpose"
    )


def test_minimum_skill_count() -> None:
    """Test that we have at least the expected number of skills."""
    skills = get_all_skills()
    # Dynamic: just verify we haven't lost skills
    assert len(skills) >= 25, f"Expected at least 25 skills, found {len(skills)}"

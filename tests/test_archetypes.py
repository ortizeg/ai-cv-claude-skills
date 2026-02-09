"""Test that all archetypes are complete."""

from __future__ import annotations

from pathlib import Path

import pytest

ARCHETYPES_DIR = Path("archetypes")


def get_all_archetypes() -> list[Path]:
    """Get all archetype directories."""
    return sorted(
        [d for d in ARCHETYPES_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )


@pytest.mark.parametrize("archetype_dir", get_all_archetypes(), ids=lambda d: d.name)
def test_archetype_has_readme(archetype_dir: Path) -> None:
    """Test archetype has README."""
    readme = archetype_dir / "README.md"
    assert readme.exists(), f"Missing README in {archetype_dir.name}"

    content = readme.read_text()
    assert len(content) > 500, f"README too short in {archetype_dir.name}"


@pytest.mark.parametrize("archetype_dir", get_all_archetypes(), ids=lambda d: d.name)
def test_archetype_has_toml(archetype_dir: Path) -> None:
    """Test archetype has archetype.toml metadata file."""
    toml_path = archetype_dir / "archetype.toml"
    assert toml_path.exists(), f"Missing archetype.toml in {archetype_dir.name}"


@pytest.mark.parametrize("archetype_dir", get_all_archetypes(), ids=lambda d: d.name)
def test_archetype_has_structure(archetype_dir: Path) -> None:
    """Test archetype has expected structure."""
    has_template = (archetype_dir / "template").exists()
    readme = archetype_dir / "README.md"
    has_structure_doc = "```" in readme.read_text() if readme.exists() else False

    assert has_template or has_structure_doc, f"Archetype {archetype_dir.name} missing structure"


def test_minimum_archetype_count() -> None:
    """Test that we have at least the expected number of archetypes."""
    archetypes = get_all_archetypes()
    assert len(archetypes) >= 6, f"Expected at least 6 archetypes, found {len(archetypes)}"

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
def test_archetype_has_structure(archetype_dir: Path) -> None:
    """Test archetype has expected structure."""
    # Should have template directory or structure documentation
    has_template = (archetype_dir / "template").exists()
    readme = archetype_dir / "README.md"
    has_structure_doc = "```" in readme.read_text() if readme.exists() else False

    assert has_template or has_structure_doc, f"Archetype {archetype_dir.name} missing structure"


def test_all_expected_archetypes_exist() -> None:
    """Test all expected archetypes exist."""
    expected = {
        "pytorch-training-project",
        "cv-inference-service",
        "research-notebook",
        "library-package",
        "data-processing-pipeline",
        "model-zoo",
    }

    actual = {d.name for d in get_all_archetypes()}

    missing = expected - actual
    assert not missing, f"Missing archetypes: {missing}"

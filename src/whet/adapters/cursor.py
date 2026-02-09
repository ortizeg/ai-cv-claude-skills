"""Cursor adapter — converts SKILL.md to Cursor rules format."""

from __future__ import annotations

from pathlib import Path

from whet.core.skill import Skill


class CursorAdapter:
    """Adapter for Cursor.

    Cursor uses .cursor/rules/ with markdown rule files.
    Strips YAML frontmatter and writes as .md rule files.
    """

    @property
    def name(self) -> str:
        return "cursor"

    def install_skill(self, skill: Skill, target_dir: Path) -> Path:
        """Write skill as a Cursor rule file."""
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / f"{skill.name}.md"

        content = skill.read_skill_md()
        # Strip YAML frontmatter for Cursor
        content = _strip_frontmatter(content)

        dest.write_text(content)
        return dest

    def remove_skill(self, skill_name: str, target_dir: Path) -> bool:
        """Remove a Cursor rule file."""
        rule_file = target_dir / f"{skill_name}.md"
        if rule_file.is_file():
            rule_file.unlink()
            return True
        return False

    def list_installed(self, target_dir: Path) -> list[str]:
        """List installed rules by scanning .md files."""
        if not target_dir.is_dir():
            return []
        return sorted(f.stem for f in target_dir.glob("*.md"))


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return content

    lines = content.split("\n")
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[i + 1 :]).lstrip("\n")

    return content

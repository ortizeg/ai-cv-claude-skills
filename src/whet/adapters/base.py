"""Base adapter protocol for platform-specific skill installation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from whet.core.skill import Skill


class PlatformAdapter(Protocol):
    """Protocol for platform-specific skill installation."""

    @property
    def name(self) -> str:
        """Platform name."""
        ...

    def install_skill(self, skill: Skill, target_dir: Path) -> Path:
        """Install a skill to the target directory.

        Returns the path where the skill was installed.
        """
        ...

    def remove_skill(self, skill_name: str, target_dir: Path) -> bool:
        """Remove an installed skill.

        Returns True if the skill was found and removed.
        """
        ...

    def list_installed(self, target_dir: Path) -> list[str]:
        """List names of installed skills."""
        ...

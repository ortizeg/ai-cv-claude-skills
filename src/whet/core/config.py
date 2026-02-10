"""Global and local configuration model."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class Platform(str, Enum):
    """Supported AI coding agent platforms."""

    CLAUDE = "claude"
    ANTIGRAVITY = "antigravity"
    CURSOR = "cursor"
    COPILOT = "copilot"


class PlatformPaths(BaseModel):
    """Platform-specific paths for skill installation."""

    global_dir: Path
    local_dir: Path


PLATFORM_PATHS: dict[Platform, PlatformPaths] = {
    Platform.CLAUDE: PlatformPaths(
        global_dir=Path.home() / ".claude" / "skills",
        local_dir=Path(".claude") / "skills",
    ),
    Platform.ANTIGRAVITY: PlatformPaths(
        global_dir=Path.home() / ".gemini" / "antigravity" / "skills",
        local_dir=Path(".agent") / "skills",
    ),
    Platform.CURSOR: PlatformPaths(
        global_dir=Path.home() / ".cursor" / "rules",
        local_dir=Path(".cursor") / "rules",
    ),
    Platform.COPILOT: PlatformPaths(
        global_dir=Path.home() / ".github",
        local_dir=Path(".github"),
    ),
}


class WhetConfig(BaseModel):
    """Whet configuration persisted to disk."""

    target: Platform = Platform.CLAUDE
    skills_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3] / "skills")
    agents_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3] / "agents")
    archetypes_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3] / "archetypes"
    )

    def get_platform_paths(self) -> PlatformPaths:
        """Get paths for the current target platform."""
        return PLATFORM_PATHS[self.target]

"""Auto-detect platform from project directory markers."""

from __future__ import annotations

from pathlib import Path

from whet.core.config import Platform

# Detection priority order
_MARKERS: list[tuple[str, Platform]] = [
    (".claude", Platform.CLAUDE),
    (".agent", Platform.ANTIGRAVITY),
    (".cursor", Platform.CURSOR),
    (".github", Platform.COPILOT),
]


def detect_platform(project_dir: Path | None = None) -> Platform | None:
    """Detect the AI platform from project directory markers.

    Checks for platform-specific directories in priority order.
    Returns None if no platform markers are found.
    """
    root = project_dir or Path.cwd()

    for marker, platform in _MARKERS:
        if (root / marker).is_dir():
            return platform

    return None

"""Settings merge and diff engine."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class SettingsTemplate(BaseModel, frozen=True):
    """A settings template with permissions."""

    permissions: dict[str, list[str]] = Field(default_factory=dict)


def load_template(template_path: Path) -> SettingsTemplate:
    """Load a settings template from a JSON file."""
    raw = json.loads(template_path.read_text())
    return SettingsTemplate(**raw)


def load_existing(settings_path: Path) -> SettingsTemplate | None:
    """Load existing settings file, or None if not found."""
    if not settings_path.exists():
        return None
    raw = json.loads(settings_path.read_text())
    return SettingsTemplate(**raw)


def merge_settings(
    existing: SettingsTemplate,
    template: SettingsTemplate,
) -> SettingsTemplate:
    """Merge template permissions into existing settings.

    - Keeps all existing allow entries
    - Appends new entries from template
    - Deduplicates the result
    """
    existing_allow = existing.permissions.get("allow", [])
    template_allow = template.permissions.get("allow", [])

    merged = list(dict.fromkeys(existing_allow + template_allow))

    return SettingsTemplate(permissions={"allow": merged})


def diff_settings(
    existing: SettingsTemplate | None,
    template: SettingsTemplate,
) -> tuple[list[str], list[str], list[str]]:
    """Compute the diff between existing and template settings.

    Returns (new_entries, existing_entries, all_merged).
    """
    existing_allow = set(existing.permissions.get("allow", [])) if existing else set()
    template_allow = set(template.permissions.get("allow", []))

    new_entries = sorted(template_allow - existing_allow)
    kept_entries = sorted(existing_allow)
    all_merged = sorted(existing_allow | template_allow)

    return new_entries, kept_entries, all_merged


def write_settings(settings_path: Path, template: SettingsTemplate) -> None:
    """Write settings to disk as formatted JSON."""
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        {"permissions": template.permissions},
        indent=2,
    )
    settings_path.write_text(content + "\n")


def get_template_path(platform: str) -> Path:
    """Get the path to a platform's settings template."""
    templates_dir = Path(__file__).resolve().parents[3] / "settings"
    template_file = templates_dir / f"{platform}.json"
    if template_file.exists():
        return template_file
    # Fall back to base template
    return templates_dir / "base.json"


def get_settings_target(platform: str, scope: str) -> Path:
    """Get the target path for settings based on platform and scope."""
    from whet.core.config import PLATFORM_PATHS, Platform

    plat = Platform(platform)
    paths = PLATFORM_PATHS[plat]

    if scope == "global":
        if plat == Platform.CLAUDE:
            return Path.home() / ".claude" / "settings.json"
        if plat == Platform.ANTIGRAVITY:
            return Path.home() / ".gemini" / "settings.json"
        if plat == Platform.CURSOR:
            return Path.home() / ".cursor" / "settings.json"
        return paths.global_dir / "settings.json"
    else:
        if plat == Platform.CLAUDE:
            return Path(".claude") / "settings.local.json"
        if plat == Platform.ANTIGRAVITY:
            return Path(".agent") / "settings.json"
        if plat == Platform.CURSOR:
            return Path(".cursor") / "settings.json"
        return paths.local_dir / "settings.json"

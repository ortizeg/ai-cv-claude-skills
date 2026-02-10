"""Install command — bulk install skills, agents, and settings to a platform."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from whet.adapters.base import PlatformAdapter
from whet.cli import app
from whet.core.config import Platform, WhetConfig
from whet.core.skill import Skill
from whet.registry.loader import discover_agents, discover_skills

console = Console()


@app.command()
def install(
    scope_global: bool = typer.Option(False, "--global", "-g", help="Install to global directory."),
    scope_local: bool = typer.Option(False, "--local", "-l", help="Install to local project."),
    category: str | None = typer.Option(None, "--category", "--cat", help="Only install category."),
    skills_only: bool = typer.Option(False, "--skills-only", help="Only install skills."),
    agents_only: bool = typer.Option(False, "--agents-only", help="Only install agents."),
    with_settings: bool = typer.Option(
        False, "--with-settings", "-s", help="Also apply settings template."
    ),
) -> None:
    """Install skills, agents, and optionally settings."""
    if not scope_global and not scope_local:
        scope_local = True

    cfg = _get_config()
    adapter = _get_adapter(cfg.target)
    paths = cfg.get_platform_paths()

    include_skills = not agents_only
    include_agents = not skills_only

    all_skills: list[Skill] = []
    all_agents: list[Skill] = []

    if include_skills:
        all_skills = discover_skills(cfg.skills_dir)
        if category:
            all_skills = [s for s in all_skills if s.category == category]

    if include_agents:
        all_agents = discover_agents(cfg.agents_dir)

    if not all_skills and not all_agents:
        console.print("[yellow]No skills or agents found to install.[/yellow]")
        raise typer.Exit(code=1)

    if scope_global:
        if all_skills:
            _install_to(adapter, all_skills, paths.global_dir, "skills", "global")
        if all_agents:
            _install_to(adapter, all_agents, paths.global_dir, "agents", "global")
        if with_settings:
            _apply_settings(cfg.target.value, "global")

    if scope_local:
        if all_skills:
            _install_to(adapter, all_skills, paths.local_dir, "skills", "local")
        if all_agents:
            _install_to(adapter, all_agents, paths.local_dir, "agents", "local")
        if with_settings:
            _apply_settings(cfg.target.value, "local")


def _install_to(
    adapter: PlatformAdapter,
    items: list[Skill],
    target_dir: Path,
    item_type: str,
    scope_label: str,
) -> None:
    """Install skills or agents to a target directory."""
    console.print(f"\n[bold]Installing {len(items)} {item_type} ({scope_label})...[/bold]")

    for item in items:
        adapter.install_skill(item, target_dir)
        console.print(f"  [green]✓[/green] {item.name}")

    console.print(
        f"\n[bold green]✓ Installed {len(items)} {item_type} to {target_dir}[/bold green]"
    )


def _apply_settings(platform: str, scope: str) -> None:
    """Apply settings template with merge support."""
    from whet.settings.engine import (
        get_settings_target,
        get_template_path,
        load_existing,
        load_template,
        merge_settings,
        write_settings,
    )

    template_path = get_template_path(platform)
    settings_path = get_settings_target(platform, scope)
    template = load_template(template_path)
    existing = load_existing(settings_path)

    if existing:
        merged = merge_settings(existing, template)
        write_settings(settings_path, merged)
        console.print(f"\n[bold green]✓ Merged settings into {settings_path}[/bold green]")
    else:
        write_settings(settings_path, template)
        console.print(f"\n[bold green]✓ Applied settings to {settings_path}[/bold green]")


def _get_config() -> WhetConfig:
    return WhetConfig.load()


def _get_adapter(platform: Platform) -> PlatformAdapter:
    from whet.adapters.antigravity import AntigravityAdapter
    from whet.adapters.claude import ClaudeAdapter
    from whet.adapters.copilot import CopilotAdapter
    from whet.adapters.cursor import CursorAdapter

    adapters: dict[Platform, type[PlatformAdapter]] = {
        Platform.CLAUDE: ClaudeAdapter,
        Platform.ANTIGRAVITY: AntigravityAdapter,
        Platform.CURSOR: CursorAdapter,
        Platform.COPILOT: CopilotAdapter,
    }
    return adapters[platform]()

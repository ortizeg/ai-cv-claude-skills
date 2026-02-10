"""Install command — bulk install skills to a platform."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from whet.adapters.base import PlatformAdapter
from whet.cli import app
from whet.core.config import Platform, WhetConfig
from whet.core.skill import Skill
from whet.registry.loader import discover_skills

console = Console()


@app.command()
def install(
    scope_global: bool = typer.Option(False, "--global", "-g", help="Install to global directory."),
    scope_local: bool = typer.Option(False, "--local", "-l", help="Install to local project."),
    category: str | None = typer.Option(None, "--category", "--cat", help="Only install category."),
) -> None:
    """Install the full skill collection."""
    if not scope_global and not scope_local:
        scope_local = True

    cfg = _get_config()
    adapter = _get_adapter(cfg.target)
    all_skills = discover_skills(cfg.skills_dir)

    if category:
        all_skills = [s for s in all_skills if s.category == category]

    if not all_skills:
        console.print("[yellow]No skills found to install.[/yellow]")
        raise typer.Exit(code=1)

    paths = cfg.get_platform_paths()

    if scope_global:
        _install_to(adapter, all_skills, paths.global_dir, "global")

    if scope_local:
        _install_to(adapter, all_skills, paths.local_dir, "local")


def _install_to(
    adapter: PlatformAdapter, skills: list[Skill], target_dir: Path, scope_label: str
) -> None:
    """Install skills to a target directory."""
    console.print(f"\n[bold]Installing {len(skills)} skills ({scope_label})...[/bold]")

    for skill in skills:
        adapter.install_skill(skill, target_dir)
        console.print(f"  [green]✓[/green] {skill.name}")

    console.print(f"\n[bold green]✓ Installed {len(skills)} skills to {target_dir}[/bold green]")


def _get_config() -> WhetConfig:
    return WhetConfig()


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

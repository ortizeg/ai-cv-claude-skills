"""Init command — scaffold a new project from an archetype."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from whet.cli import app
from whet.core.config import WhetConfig
from whet.scaffold.engine import (
    ScaffoldContext,
    discover_archetypes,
    load_archetype,
    render_template,
)

console = Console()


@app.command(name="init")
def init_project(
    archetype_name: str | None = typer.Argument(  # noqa: UP007
        None, help="Archetype to scaffold from."
    ),
    output_dir: str | None = typer.Option(  # noqa: UP007
        None, "--output", "-o", help="Output directory (default: current dir)."
    ),
    project_name: str | None = typer.Option(  # noqa: UP007
        None, "--name", "-n", help="Project name."
    ),
    author: str | None = typer.Option(  # noqa: UP007
        None, "--author", "-a", help="Author name."
    ),
) -> None:
    """Scaffold a new project from an archetype template.

    Without arguments, lists available archetypes.
    """
    cfg = WhetConfig.load()

    if archetype_name is None:
        _list_archetypes(cfg)
        return

    archetype = load_archetype(cfg.archetypes_dir / archetype_name)
    if not archetype:
        console.print(f"[red]Archetype '{archetype_name}' not found.[/red]")
        console.print("[dim]Run 'whet init' to see available archetypes.[/dim]")
        raise typer.Exit(code=1)

    name = project_name or archetype_name
    dest = Path(output_dir) if output_dir else Path.cwd() / name

    if dest.exists() and any(dest.iterdir()):
        console.print(f"[red]Directory '{dest}' already exists and is not empty.[/red]")
        raise typer.Exit(code=1)

    context = ScaffoldContext(
        project_name=name,
        description=archetype.metadata.description,
        author=author or "",
    )

    console.print(f"[bold]Scaffolding: {archetype.metadata.name}[/bold]\n")
    console.print(f"  Project:   {context.project_name}")
    console.print(f"  Package:   {context.package_name}")
    console.print(f"  Directory: {dest}")
    console.print()

    render_template(archetype, dest, context)

    console.print(f"[bold green]✓ Project scaffolded at {dest}[/bold green]\n")

    # Show required skills
    if archetype.skills.required:
        console.print("[bold]Required skills:[/bold]")
        for skill in archetype.skills.required:
            console.print(f"  - {skill}")
        console.print(
            f"\nRun: [bold]cd {dest.name} && whet add {' '.join(archetype.skills.required)}[/bold]"
        )

    if archetype.skills.recommended:
        console.print("\n[bold]Recommended skills:[/bold]")
        for skill in archetype.skills.recommended:
            console.print(f"  - {skill}")


def _list_archetypes(cfg: WhetConfig) -> None:
    """List available archetypes."""
    archetypes = discover_archetypes(cfg.archetypes_dir)

    if not archetypes:
        console.print("[dim]No archetypes found.[/dim]")
        raise typer.Exit()

    table = Table(title="Available Archetypes")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Category", style="green")
    table.add_column("Description")
    table.add_column("Template", style="dim")

    for arch in archetypes:
        has_tmpl = "yes" if arch.has_template else "minimal"
        table.add_row(
            arch.metadata.name,
            arch.metadata.category,
            arch.metadata.description,
            has_tmpl,
        )

    console.print(table)
    console.print("\n[dim]Usage: whet init <archetype-name>[/dim]")

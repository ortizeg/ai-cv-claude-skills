"""Update command — update whet to the latest version."""

from __future__ import annotations

import subprocess
import sys

from rich.console import Console

from whet import __version__
from whet.cli import app

console = Console()


@app.command()
def update() -> None:
    """Update whet to the latest version."""
    console.print(f"[bold]Current version:[/bold] v{__version__}\n")
    console.print("Checking for updates...")

    try:
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pip", "install", "--upgrade", "whet"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            if "already satisfied" in result.stdout.lower():
                console.print("[green]✓ Already up to date.[/green]")
            else:
                console.print("[bold green]✓ Updated successfully.[/bold green]")
                console.print("[dim]Restart your shell to use the new version.[/dim]")
        else:
            console.print("[yellow]Could not update via pip. Try:[/yellow]")
            console.print("  [bold]uv tool upgrade whet[/bold]")
            console.print("  or")
            console.print("  [bold]uvx --upgrade whet list[/bold]")
    except FileNotFoundError:
        console.print("[yellow]pip not found. Try:[/yellow]")
        console.print("  [bold]uv tool upgrade whet[/bold]")

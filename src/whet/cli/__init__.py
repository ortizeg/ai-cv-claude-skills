"""whet CLI — Sharpen your AI coder."""

from __future__ import annotations

import typer
from rich.console import Console

from whet import __version__

console = Console()

app = typer.Typer(
    name="whet",
    help="Sharpen your AI coder — install expert skills into AI coding agents.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)

BANNER = r"""
  [bold cyan]██╗    ██╗██╗  ██╗███████╗████████╗[/bold cyan]
  [bold cyan]██║    ██║██║  ██║██╔════╝╚══██╔══╝[/bold cyan]
  [bold cyan]██║ █╗ ██║███████║█████╗     ██║[/bold cyan]
  [bold cyan]██║███╗██║██╔══██║██╔══╝     ██║[/bold cyan]
  [bold cyan]╚███╔███╔╝██║  ██║███████╗   ██║[/bold cyan]
  [bold cyan] ╚══╝╚══╝ ╚═╝  ╚═╝╚══════╝   ╚═╝[/bold cyan]
  [dim]sharpen your AI coder[/dim]
"""


def version_callback(value: bool) -> None:  # noqa: FBT001
    """Print version and exit."""
    if value:
        console.print(BANNER)
        console.print(f"  [bold]v{__version__}[/bold] · claude · antigravity · cursor\n")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(  # noqa: UP007
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """whet — Sharpen your AI coder.

    Install expert skills into AI coding agents (Claude Code, Antigravity, Cursor, Copilot).
    """


# Import and register sub-commands
from whet.cli import config as _config_mod  # noqa: E402, F401
from whet.cli import doctor as _doctor_mod  # noqa: E402, F401
from whet.cli import init as _init_mod  # noqa: E402, F401
from whet.cli import install as _install_mod  # noqa: E402, F401
from whet.cli import settings as _settings_mod  # noqa: E402, F401
from whet.cli import skills as _skills_mod  # noqa: E402, F401
from whet.cli import target as _target_mod  # noqa: E402, F401
from whet.cli import update as _update_mod  # noqa: E402, F401

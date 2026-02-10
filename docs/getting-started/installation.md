# Installation

This guide walks you through setting up whet and the AI/CV skills framework.

## Quick Install

The fastest way to use whet is via `uvx` (no install needed):

```bash
# List available skills
uvx whet list

# Install skills to Claude Code
uvx whet install --global
```

Or install permanently:

```bash
uv tool install whet
```

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| [uv](https://docs.astral.sh/uv/) | Latest | Python package management |
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | Latest | The CLI that loads and uses skills |
| [Git](https://git-scm.com/) | >= 2.30 | Version control |
| Python | >= 3.11 | Runtime (managed by uv) |

### Installing uv

uv is the recommended package manager for whet. It handles Python versions, dependencies, and tool installation.

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:

```bash
uv --version
```

### Installing Claude Code

Claude Code is Anthropic's official CLI for Claude. Install it via npm:

```bash
npm install -g @anthropic-ai/claude-code
```

Verify:

```bash
claude --version
```

## Using whet

### Install Skills Globally

Install all skills to your AI coding agent:

```bash
# Target Claude Code (default)
whet install --global

# Target Google Antigravity
whet target antigravity
whet install --global

# Target Cursor
whet target cursor
whet install --global
```

### Add Skills to a Project

Install specific skills into your current project:

```bash
whet add pytorch-lightning wandb hydra-config
```

### Browse Skills

```bash
# List all skills
whet list

# Search by keyword
whet search training

# Show skill details
whet info pytorch-lightning
```

## Development Setup

If you plan to contribute to whet or run the tests:

```bash
git clone https://github.com/ortizeg/whet.git
cd whet
uv sync --all-extras
```

### Running Checks

```bash
# Run all tests
just test

# Lint
just lint

# Type check
just typecheck

# Run everything
just check
```

Or directly with uv:

```bash
uv run pytest tests/ -v
uv run ruff check .
uv run mypy src/whet/ tests/ --strict
```

## Verifying the Installation

Run the completeness test to confirm all skills are properly structured:

```bash
uv run pytest tests/test_skills_completeness.py -v
```

This test verifies that:

- All skills have YAML frontmatter in SKILL.md
- Each skill has `SKILL.md`, `README.md`, and `skill.toml`
- `SKILL.md` files contain substantial content (500+ characters)
- `SKILL.md` files include code examples and headers
- `README.md` files explain the skill's purpose

## Supported Platforms

| Platform | Skill Format | Global Directory |
|----------|-------------|-----------------|
| **Claude Code** | SKILL.md (native) | `~/.claude/skills/` |
| **Google Antigravity** | SKILL.md (native) | `~/.gemini/antigravity/skills/` |
| **Cursor** | .md rules (converted) | `~/.cursor/rules/` |
| **GitHub Copilot** | Single aggregated file | `~/.github/` |

## Next Steps

With whet installed, proceed to the [Quick Start](quick-start.md) guide to create your first AI-assisted CV project.

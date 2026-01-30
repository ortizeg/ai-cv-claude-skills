# Installation

This guide walks you through setting up the AI/CV Claude Skills framework on your machine.

## Prerequisites

Before you begin, ensure you have the following installed:

| Tool | Version | Purpose |
|------|---------|---------|
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | Latest | The CLI that loads and uses skills |
| [Pixi](https://pixi.sh/) | >= 0.30 | Package and environment management |
| [Git](https://git-scm.com/) | >= 2.30 | Version control |
| Python | >= 3.10 | Runtime (managed by Pixi) |

### Installing Claude Code

Claude Code is Anthropic's official CLI for Claude. Install it via npm:

```bash
npm install -g @anthropic-ai/claude-code
```

Verify the installation:

```bash
claude --version
```

### Installing Pixi

Pixi is the recommended package manager for this framework. It handles Python versions, CUDA dependencies, and reproducible environments.

```bash
# macOS / Linux
curl -fsSL https://pixi.sh/install.sh | bash

# Windows (PowerShell)
iwr -useb https://pixi.sh/install.ps1 | iex
```

Verify:

```bash
pixi --version
```

## Cloning the Repository

```bash
git clone https://github.com/ortizeg/ai-cv-claude-skills.git
cd ai-cv-claude-skills
```

## Repository Layout

After cloning, you will see this structure:

```
ai-cv-claude-skills/
    skills/              # 21 skill modules (each with SKILL.md, README.md)
    agents/              # 4 agent persona definitions
    archetypes/          # 6 project archetype templates
    docs/                # Documentation (you are reading it)
    tests/               # Validation tests for skills
    mkdocs.yml           # Documentation site configuration
```

## Configuring Claude Code

The skills framework works by providing Claude Code with context files. There are two ways to use it:

### Option 1: Reference Skills Directly

Point Claude Code at skill files when starting a session:

```bash
# Load a specific skill
claude --skill ./skills/pytorch-lightning/SKILL.md

# Load multiple skills
claude --skill ./skills/pytorch-lightning/SKILL.md \
       --skill ./skills/hydra-config/SKILL.md \
       --skill ./skills/wandb/SKILL.md
```

### Option 2: Copy Skills Into Your Project

For persistent use in a project, copy the relevant skill files into your project's `.claude/` directory:

```bash
# Create the Claude configuration directory in your project
mkdir -p /path/to/your/project/.claude/skills

# Copy the skills you need
cp skills/pytorch-lightning/SKILL.md /path/to/your/project/.claude/skills/pytorch-lightning.md
cp skills/hydra-config/SKILL.md /path/to/your/project/.claude/skills/hydra-config.md
```

### Option 3: Symlink the Skills Directory

For development workflows where you want the latest skill updates:

```bash
ln -s /path/to/ai-cv-claude-skills/skills /path/to/your/project/.claude/skills
```

## Setting Up a Development Environment

If you plan to contribute to the framework itself or run the tests:

```bash
cd ai-cv-claude-skills

# Create environment with Pixi
pixi install

# Run the test suite
pixi run pytest tests/
```

## Verifying the Installation

Run the completeness test to confirm all skills are properly structured:

```bash
pixi run pytest tests/test_skills_completeness.py -v
```

This test verifies that:

- All 21 expected skills exist
- Each skill has both a `SKILL.md` and `README.md`
- `SKILL.md` files contain substantial content (500+ characters)
- `SKILL.md` files include code examples and headers
- `README.md` files explain the skill's purpose

## Next Steps

With the framework installed, proceed to the [Quick Start](quick-start.md) guide to create your first AI-assisted CV project.

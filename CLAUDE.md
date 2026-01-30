# CLAUDE.md

This file provides context for Claude Code when working in this repository.

## Project Overview

**AI/CV Claude Skills** is a production-ready skills framework for computer vision and deep learning projects, designed for Claude Code. It provides:

- **25 Skills** — Best-practice knowledge modules for specific tools and patterns (PyTorch Lightning, Pydantic, Docker, etc.)
- **4 Agents** — Pre-configured behavioral profiles (Expert Coder, ML Engineer, Code Review, Test Engineer)
- **6 Archetypes** — Complete project templates for common CV/ML project types

### Repository Structure

```
ai-cv-claude-skills/
├── skills/              # 25 individual skill definitions (each has SKILL.md + README.md)
├── agents/              # 4 agent definitions (each has SKILL.md + README.md + action.yml)
├── archetypes/          # 6 project templates (each has README.md + template/)
├── docs/                # MkDocs Material documentation
├── tests/               # Self-tests validating completeness
├── .github/workflows/   # CI/CD (test, lint, docs)
├── pixi.toml            # Development environment and tasks
├── pyproject.toml       # Tool configuration (ruff, mypy, pytest)
└── mkdocs.yml           # Documentation site config
```

## Development Commands

```bash
pixi run test              # Run all tests
pixi run lint              # Ruff linting
pixi run typecheck         # MyPy strict type checking
pixi run ruff format --check .  # Check formatting
pixi run docs-build        # Build documentation site
pixi run docs-serve        # Serve docs locally
```

## How to Add a New Skill

1. Create `skills/<name>/SKILL.md` — Must be >500 chars, include code examples (fenced blocks) and headers
2. Create `skills/<name>/README.md` — Must explain purpose (include words like "when", "use", "purpose")
3. Create `docs/skills/<name>.md` — Documentation page following existing pattern (Purpose, When to Use, Key Patterns, Anti-Patterns, Combines Well With)
4. Add `"<name>"` to `tests/test_skills_completeness.py` expected set
5. Add nav entry to `mkdocs.yml` under Skills section (in appropriate category position)
6. Add row to `docs/skills/index.md` in the appropriate category table
7. Update skill counts — see "Files with Hardcoded Counts" below

## How to Add a New Agent

1. Create `agents/<name>/SKILL.md` — Must be >500 chars
2. Create `agents/<name>/README.md` — Agent overview
3. If blocking agent: create `agents/<name>/action.yml`
4. Create `docs/agents/<name>.md` — Documentation page
5. Add `"<name>"` to `tests/test_agents.py` expected set
6. Add nav entry to `mkdocs.yml` under Agents section
7. Update agent counts in README.md and docs/index.md

## How to Add a New Archetype

1. Create `archetypes/<name>/README.md` — Must be >500 chars, include code blocks
2. Create `archetypes/<name>/template/` directory with template files
3. Create `docs/archetypes/<name>.md` — Documentation page
4. Add `"<name>"` to `tests/test_archetypes.py` expected set
5. Add nav entry to `mkdocs.yml` under Archetypes section
6. Update archetype counts in README.md and docs/index.md

## Files with Hardcoded Counts

These files contain hardcoded skill/agent/archetype counts that **must** be updated when adding or removing skills, agents, or archetypes:

### Skill count (currently 25)

- `README.md` — lines 13, 64, 71
- `docs/index.md` — lines 9, 15, 48, 79

### Agent count (currently 4)

- `README.md` — line 14
- `docs/index.md` — line 16

### Archetype count (currently 6)

- `README.md` — line 15
- `docs/index.md` — line 17

## Conventions

- **Python 3.11+** — minimum version, use modern syntax (PEP 604 unions, etc.)
- **Ruff** — linting and formatting (line-length 100, rules: E, F, I, N, UP, S, B, A, C4, T20, SIM)
- **MyPy strict** — all code must pass `mypy --strict`
- **Pixi** — dependency and environment management (never use pip directly)
- **Loguru** — mandatory logging convention across all skills
- **Pydantic V2** — `BaseModel` for all configs and data structures
- **src-layout** — mandatory for all project archetypes

## CI/CD

Three GitHub Actions workflows, all must pass before merge:

1. **test.yml** — `pixi run test-cov`, `pixi run lint`, `pixi run typecheck`
2. **lint.yml** — `ruff format --check .`, linting, type checking, YAML validation
3. **docs.yml** — Builds and deploys MkDocs to GitHub Pages (main branch only)

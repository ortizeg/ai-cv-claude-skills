# Pre-commit Hooks Skill

## Purpose

This skill provides guidance on configuring and using pre-commit hooks for Python and machine learning projects. Pre-commit automates code quality checks before every commit, catching formatting issues, lint errors, type problems, and accidental secret or large file commits early in the development cycle.

## Usage

Reference this skill when:

- Setting up a new Python or ML project repository.
- Adding or configuring pre-commit hooks.
- Integrating Ruff (linting and formatting) into the development workflow.
- Configuring MyPy for static type checking in pre-commit.
- Writing custom hooks for project-specific checks.
- Setting up CI enforcement of pre-commit hooks.
- Troubleshooting hook failures or performance issues.

## What This Skill Covers

- Installation and setup of pre-commit.
- Complete `.pre-commit-config.yaml` for Python/ML projects.
- Ruff integration for linting and formatting.
- MyPy integration for static type checking.
- Custom hooks (no large files, no secrets, config validation).
- Manual execution and hook management commands.
- GitHub Actions CI integration.
- Troubleshooting common issues.

## Benefits

- Enforces consistent code style across all contributors without manual effort.
- Catches errors before they reach CI, saving pipeline minutes.
- Prevents accidental commits of secrets, model files, and merge conflicts.
- Produces cleaner code reviews focused on logic rather than formatting.
- Provides reproducible checks with pinned hook versions.

## Key Configuration

The core configuration lives in `.pre-commit-config.yaml` at the repository root and includes hooks from `pre-commit-hooks`, `ruff-pre-commit`, and `mirrors-mypy`. See `SKILL.md` for the full configuration and detailed explanations.

# Pixi Skill

The Pixi Skill standardizes environment management and task orchestration using pixi, a fast, cross-platform package manager built on conda-forge. Unlike pip-only workflows that struggle with system-level dependencies common in CV/ML projects (CUDA toolkits, OpenCV with codec support, FFmpeg), pixi resolves both Python and system packages from conda-forge in a single lockfile. This skill defines how to structure `pixi.toml`, manage dependencies, define reusable tasks, and ensure reproducible environments across development machines, CI runners, and production containers.

Pixi replaces the patchwork of pyenv, virtualenv, pip-tools, and Makefiles with a single tool and configuration file. The `pixi.toml` file declares the project metadata, platform targets, dependency specifications with version pins, feature groups for optional capabilities (e.g., a `cuda` feature, a `dev` feature), and task definitions that serve as the project's command interface. Every operation -- training, testing, linting, building Docker images -- is exposed as a `pixi run <task>` command, giving the team a consistent entry point regardless of platform.

## When to Use

- At project initialization to establish the environment and dependency management strategy.
- When a project requires system-level dependencies (CUDA, OpenCV, FFmpeg) alongside Python packages.
- When you need reproducible environments across macOS (development) and Linux (CI/production).
- When replacing scattered Makefile targets or shell scripts with a structured task runner.

## Key Features

- **Conda-forge resolution** -- resolves Python packages, CUDA toolkits, and system libraries from a single channel with a unified lockfile.
- **pixi.toml configuration** -- single file declaring project metadata, platforms, dependencies, features, and tasks.
- **Task definitions** -- named commands with dependency chains (`pixi run train`, `pixi run lint`, `pixi run test`) replacing Makefiles.
- **Multi-platform support** -- targets `linux-64`, `osx-arm64`, and `osx-64` with platform-specific dependency overrides.
- **Feature groups** -- optional dependency sets (e.g., `[feature.cuda.dependencies]`, `[feature.dev.dependencies]`) activated per environment.
- **Lockfile pinning** -- `pixi.lock` captures exact resolved versions for fully reproducible installs across machines and CI.

## Related Skills

- **[Code Quality](../code-quality/)** -- pixi tasks wrap Ruff and mypy commands for consistent invocation.
- **[Docker CV](../docker-cv/)** -- Docker builds use pixi to install dependencies inside containers, ensuring parity with local environments.
- **[GitHub Actions](../github-actions/)** -- CI workflows call `pixi run` tasks rather than raw commands, keeping CI and local workflows identical.
- **[Master Skill](../master-skill/)** -- project scaffolding generates the initial `pixi.toml` based on selected archetype and skills.

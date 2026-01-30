# GitHub Actions Skill

The GitHub Actions Skill defines CI/CD workflow patterns tailored to machine learning projects, where pipelines must handle not just code quality and tests but also model training validation, artifact management, and GPU-accelerated workloads. Standard web-application CI patterns break down in ML contexts: test suites need GPU runners for integration tests, artifacts include large model checkpoints, and validation involves metrics thresholds rather than simple pass/fail assertions. This skill provides workflow templates that address these ML-specific concerns while maintaining fast feedback on code quality.

The skill structures workflows into tiers: a fast tier for linting, formatting, and type checking that runs on every push; a medium tier for unit tests and lightweight model smoke tests that runs on pull requests; and a heavy tier for full training validation and integration tests that runs on merge to main or on manual dispatch. Matrix strategies cover Python versions and platform combinations, caching is configured for pixi environments and pip downloads, and artifact actions manage model checkpoints and evaluation reports. The skill also integrates with code-review and test-engineer AI agents for automated PR feedback.

## When to Use

- When setting up CI/CD for a new ML project repository.
- When adding GPU-accelerated testing to an existing pipeline.
- When configuring automated model validation that gates deployment on metrics thresholds.
- When integrating AI-powered code review or test generation agents into the PR workflow.

## Key Features

- **Tiered workflow structure** -- fast (lint/format/type-check), medium (unit tests), and heavy (training validation) tiers with appropriate triggers.
- **Matrix testing** -- strategies covering Python versions, operating systems, and optional CUDA/CPU variants.
- **Pixi environment caching** -- caches `~/.pixi` and lockfile hashes for fast, reproducible CI environments.
- **GPU runner support** -- self-hosted runner labels and configuration for NVIDIA GPU-equipped CI machines.
- **Artifact management** -- upload/download actions for model checkpoints, evaluation reports, and training logs.
- **Agent integration** -- workflow steps that trigger code-review and test-engineer agents for automated PR feedback.

## Related Skills

- **[Code Quality](../code-quality/)** -- the fast CI tier runs Ruff and mypy checks defined by the Code Quality skill.
- **[Pixi](../pixi/)** -- CI workflows use `pixi run` tasks, keeping CI commands identical to local development commands.
- **[Docker CV](../docker-cv/)** -- heavy-tier workflows build and push Docker images as part of the release pipeline.
- **[VS Code](../vscode/)** -- debug configurations for locally reproducing CI failures with the same environment and commands.

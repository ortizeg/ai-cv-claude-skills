# Agents Overview

The ai-cv-claude-skills framework includes four specialized agents that assist with different aspects of ML/CV development. Agents are divided into two categories based on their enforcement level.

## Agent Summary

| Agent | Role | Strictness | Deployment |
|-------|------|------------|------------|
| [Expert Coder](expert-coder.md) | Primary coding assistant | Advisory | Claude Code |
| [ML Engineer](ml-engineer.md) | Architecture and training guidance | Advisory | Claude Code |
| [Code Review](code-review.md) | Automated quality enforcement | **Blocking** | GitHub Action |
| [Test Engineer](test-engineer.md) | Testing and coverage enforcement | **Blocking** | GitHub Action |

## Advisory vs Blocking

**Advisory agents** guide development but do not prevent actions. They suggest improvements, explain rationale, and generate code following standards. The Expert Coder and ML Engineer agents operate in this mode through Claude Code.

**Blocking agents** run in CI and must pass before a pull request can be merged. The Code Review and Test Engineer agents operate as GitHub Actions that gate merges on code quality, type safety, test coverage, and security checks.

## How They Work Together

1. **During development**, the Expert Coder generates code following project patterns (abstractions, Pydantic, type hints).
2. **For ML decisions**, the ML Engineer reviews model architecture, training setup, and experiment configuration.
3. **On push/PR**, the Code Review agent validates formatting, linting, types, and security.
4. **On push/PR**, the Test Engineer runs the full test suite and enforces coverage thresholds.

This layered approach provides guidance during development and enforcement during integration.

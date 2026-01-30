# Agents

This directory contains 4 specialized agents for AI/CV projects.

## Agent Overview

| Agent | Role | Strictness | Deployment |
|-------|------|------------|------------|
| **Expert Coder** | Primary coding assistant | Advisory | Claude Code |
| **ML Engineer** | ML architecture and training guidance | Advisory | Claude Code |
| **Code Review** | Automated quality enforcement | Blocking | GitHub Action |
| **Test Engineer** | Testing and coverage enforcement | Blocking | GitHub Action |

## Advisory vs Blocking

- **Advisory** agents guide and suggest but do not prevent actions
- **Blocking** agents run in CI and must pass before merging

## Agent Files

Each agent contains:
- `SKILL.md` — Instructions for Claude Code
- `README.md` — Human-readable documentation
- `action.yml` — GitHub Action definition (blocking agents only)
- `examples/` — Example usage sessions

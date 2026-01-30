# AI/CV Claude Skills

[![Tests](https://github.com/ortizeg/ai-cv-claude-skills/actions/workflows/test.yml/badge.svg)](https://github.com/ortizeg/ai-cv-claude-skills/actions/workflows/test.yml)
[![Docs](https://github.com/ortizeg/ai-cv-claude-skills/actions/workflows/docs.yml/badge.svg)](https://ortizeg.github.io/ai-cv-claude-skills/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-ready skills framework for computer vision and deep learning projects using Claude Code.

## What is This?

A comprehensive, prescriptive framework providing:

- **21 Skills**: Best practices for PyTorch, Pydantic, Docker, testing, and more
- **4 Specialized Agents**: Code review, testing, ML engineering, and expert coding
- **6 Project Archetypes**: Ready-to-use templates for common CV/ML workflows
- **Enforced Standards**: Type safety, code quality, and testing requirements

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ortizeg/ai-cv-claude-skills.git

# Create a new project with Claude Code
claude

You: "Using ai-cv-claude-skills, create a new pytorch-training-project for object detection"
```

## Core Principles

1. **Abstraction First** — Wrap external libraries (VideoReader, ImageLoader, etc.)
2. **Pydantic Everywhere** — Use Pydantic V2 BaseModel for configs and data structures
3. **Type Safety** — Full type hints with mypy strict mode, no `Any` unless unavoidable
4. **Prescriptive Standards** — Enforce configurations, block violations in CI
5. **Testing Required** — 80%+ coverage, comprehensive test suites

## Features

### Prescriptive Standards
- Full type hints (mypy strict mode)
- Pydantic V2 for all configs and data structures
- 80%+ test coverage requirement
- Ruff formatting and linting

### Four Agents
| Agent | Role | Strictness |
|-------|------|------------|
| **Expert Coder** | Primary development assistant | Advisory |
| **ML Engineer** | Architecture and training guidance | Advisory |
| **Code Review** | Automated quality checks | Blocking |
| **Test Engineer** | Coverage enforcement | Blocking |

### Six Archetypes
| Archetype | Use Case |
|-----------|----------|
| PyTorch Training Project | Model training with Lightning, Hydra, experiment tracking |
| CV Inference Service | FastAPI + ONNX deployment |
| Research Notebook | Jupyter-based experimentation |
| Library Package | Reusable Python package for PyPI |
| Data Processing Pipeline | ETL workflows for datasets |
| Model Zoo | Collection of pretrained models |

### 21 Skills
Skills cover PyTorch Lightning, Pydantic, code quality, Docker, Hydra, testing, OpenCV, visualization, packaging, CI/CD, editor config, pre-commit hooks, experiment tracking (W&B, MLflow, TensorBoard), data versioning (DVC), ONNX export, abstraction patterns, and library evaluation.

## Repository Structure

```
ai-cv-claude-skills/
├── skills/          # 21 individual skill definitions
├── agents/          # 4 agent definitions
├── archetypes/      # 6 project templates
├── docs/            # MkDocs Material documentation
├── tests/           # Self-tests for the skills repo
├── pixi.toml        # Development environment
├── pyproject.toml   # Tool configuration
└── .github/         # CI/CD workflows
```

## Documentation

Full documentation: [https://ortizeg.github.io/ai-cv-claude-skills/](https://ortizeg.github.io/ai-cv-claude-skills/)

- [Getting Started](https://ortizeg.github.io/ai-cv-claude-skills/getting-started/installation/)
- [Skills Reference](https://ortizeg.github.io/ai-cv-claude-skills/skills/)
- [Agents Guide](https://ortizeg.github.io/ai-cv-claude-skills/agents/)
- [Project Archetypes](https://ortizeg.github.io/ai-cv-claude-skills/archetypes/)

## Development

```bash
# Install dependencies
pixi install

# Run tests
pixi run test

# Run linting
pixi run lint

# Build documentation
pixi run docs-serve
```

## Contributing

Contributions welcome! See [Adding Skills Guide](docs/guides/adding-skills.md) for how to add new skills.

## License

MIT License — see [LICENSE](LICENSE) file.

## Author

Enrique G. Ortiz ([@ortizeg](https://github.com/ortizeg))

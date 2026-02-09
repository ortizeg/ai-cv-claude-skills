# VS Code

The VS Code skill configures Visual Studio Code for productive ML/CV development, including extensions, settings, debug configurations, and workspace setup.

**Skill directory:** `skills/vscode/`

## Purpose

A well-configured editor eliminates friction. This skill teaches Claude Code to generate VS Code configuration files that enable Python IntelliSense with ML libraries, configure debugging for training scripts with Hydra arguments, set up remote container development, and integrate linting and formatting on save.

## When to Use

- Setting up a new ML project's development environment
- Configuring debugging for PyTorch/Lightning training scripts
- Setting up remote development with Docker Dev Containers
- Standardizing editor settings across a team

## Key Patterns

### Workspace Settings

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": ".pixi/envs/default/bin/python",
    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.autoImportCompletions": true,
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pixi": true,
        "**/outputs": true,
        "**/*.egg-info": true
    }
}
```

### Debug Configuration

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Train (Debug)",
            "type": "debugpy",
            "request": "launch",
            "module": "my_project.train",
            "args": ["trainer.max_epochs=2", "trainer.fast_dev_run=true"],
            "cwd": "${workspaceFolder}",
            "env": {"HYDRA_FULL_ERROR": "1"}
        },
        {
            "name": "Test (Debug)",
            "type": "debugpy",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v", "-x"],
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

### Recommended Extensions

```json
// .vscode/extensions.json
{
    "recommendations": [
        "charliermarsh.ruff",
        "ms-python.python",
        "ms-python.debugpy",
        "ms-toolsai.jupyter",
        "redhat.vscode-yaml",
        "ms-vscode-remote.remote-containers"
    ]
}
```

## Anti-Patterns to Avoid

- Do not commit user-specific settings (font size, theme) to `.vscode/settings.json`
- Do not hardcode absolute paths in launch configurations
- Avoid conflicting formatters -- use ruff as the single Python formatter
- Do not rely on VS Code for type checking if mypy is the project standard -- configure both consistently

## Combines Well With

- **Pixi** -- Point VS Code at the Pixi-managed Python interpreter
- **Code Quality** -- Ruff integration for format-on-save
- **Docker CV** -- Dev Container configuration for containerized development
- **Hydra Config** -- Debug launch configs with Hydra overrides

## Full Reference

See [`skills/vscode/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/vscode/SKILL.md) for patterns including Dev Container configuration, multi-root workspaces, and task definitions for common ML workflows.

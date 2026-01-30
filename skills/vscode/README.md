# VS Code Skill

The VS Code Skill standardizes editor configuration for computer vision and machine learning development, ensuring that every team member has a consistent, productive development experience from the moment they open the project. ML development involves unique workflows -- debugging training loops, profiling GPU utilization, remote development on GPU servers, and navigating large codebases with deeply nested module hierarchies. This skill configures VS Code workspace settings, recommended extensions, debug launch configurations, and remote development profiles to support all of these workflows out of the box.

The skill generates a `.vscode/` directory containing `settings.json` for workspace configuration, `extensions.json` for recommended extension installs, and `launch.json` for debug configurations. Workspace settings configure the Python interpreter to use the pixi-managed environment, enable Ruff and mypy integration for inline diagnostics, set up test discovery for pytest, and configure file associations and exclusions appropriate for ML projects (ignoring checkpoints, datasets, and large binary files in the explorer). Debug configurations cover the three most common ML debugging scenarios: stepping through a training run, running a specific test with breakpoints, and debugging a model serving endpoint.

## When to Use

- At project initialization to establish a shared editor configuration for the team.
- When onboarding a new developer who needs to be productive immediately without manual editor setup.
- When configuring remote development on GPU-equipped servers via SSH or Dev Containers.
- When adding debug configurations for new training scripts, test suites, or serving endpoints.

## Key Features

- **Workspace settings** -- Python interpreter path, Ruff formatting on save, mypy diagnostics, and pytest discovery configured automatically.
- **Recommended extensions** -- curated list including Python, Pylance, Ruff, Docker, Remote SSH, and Jupyter with one-click install.
- **Training debug config** -- launch configuration for stepping through Lightning Trainer runs with breakpoints in training/validation steps.
- **Test debug config** -- pytest launch configuration with configurable markers, verbosity, and fixture debugging support.
- **Serving debug config** -- FastAPI/Flask debug configuration for model serving endpoints with hot reload.
- **Remote development** -- SSH and Dev Container profiles for GPU server development with port forwarding for TensorBoard and experiment trackers.

## Related Skills

- **[Code Quality](../code-quality/)** -- VS Code settings enable Ruff and mypy extensions that surface code quality issues inline.
- **[Pixi](../pixi/)** -- the workspace Python interpreter is configured to point to the pixi-managed environment.
- **[PyTorch Lightning](../pytorch-lightning/)** -- debug launch configurations target Lightning training scripts with correct working directory and arguments.
- **[Docker CV](../docker-cv/)** -- Dev Container configurations reuse the project's Dockerfile for consistent containerized development.

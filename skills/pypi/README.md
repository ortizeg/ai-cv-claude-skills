# PyPI Publishing Skill

## Purpose

This skill provides a complete workflow for packaging and publishing Python projects to PyPI. It covers `pyproject.toml` configuration with both Hatchling and Setuptools backends, version management, src layout, entry points, building, publishing, trusted publishers via GitHub Actions, and release automation.

## When to Use

- You are publishing a Python package (library, CLI tool, or model wrapper) to PyPI
- You need to set up `pyproject.toml` for a new project
- You want to automate releases with GitHub Actions and trusted publishers
- You need to configure entry points for CLI commands or plugin systems
- You are setting up version management with git tags

## Key Patterns

- **src layout**: `src/my_package/` structure to prevent local import accidents
- **Hatchling build backend**: Modern, fast, with good defaults
- **Trusted publishers**: OIDC authentication between GitHub Actions and PyPI (no API tokens)
- **setuptools-scm / hatch-vcs**: Derive version automatically from git tags
- **Multi-stage release workflow**: Build, test install, publish to TestPyPI, publish to PyPI, create GitHub release

## Usage

```bash
# Build the package
python -m build

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*

# Or just push a tag and let GitHub Actions handle it
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

## Benefits

- Makes your package installable with a simple `pip install`
- Trusted publishers eliminate API token management
- Automated release workflow reduces human error
- Proper src layout prevents "works on my machine" issues
- Entry points provide clean CLI integration

## See Also

- `SKILL.md` in this directory for full documentation and code examples
- `github-actions` skill for CI/CD workflow patterns

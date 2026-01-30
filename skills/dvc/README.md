# Data Version Control (DVC) Skill

## Purpose

This skill provides guidance on using DVC to version control datasets, models, and ML pipelines that are too large for Git. DVC extends Git with lightweight metafiles while storing actual data on configurable remote storage (S3, GCS, Azure, local). It is opt-in and should be used when projects involve large binary files.

## Usage

Reference this skill when:

- Tracking large datasets or model files that cannot go in Git.
- Setting up remote storage for team data sharing.
- Defining reproducible ML pipelines with `dvc.yaml`.
- Versioning data and models alongside code using Git tags.
- Running and comparing experiments with different parameters.
- Onboarding new team members who need project data.
- Managing cache and storage for large datasets.

## Opt-in Nature

DVC is never a hard requirement. Projects should:

- Document the DVC setup in the project README.
- Provide alternative data download scripts for users without DVC.
- Keep DVC configuration files (`.dvc/config`) in version control.

## Setup

```bash
pip install dvc[s3]    # Install with S3 support
dvc init               # Initialize in a Git repo
dvc remote add -d myremote s3://bucket/path
```

## What This Skill Covers

- Installation and initialization.
- Tracking files and directories with `dvc add`.
- Remote storage configuration (S3, GCS, Azure, SSH, local).
- Pipeline definitions in `dvc.yaml` with stages, deps, and outputs.
- Data and model versioning with Git tags.
- Experiment tracking and comparison.
- Git integration and collaboration workflows.
- Large dataset management and cache maintenance.
- Python API for programmatic access.

## Benefits

- Full version history for datasets and models, tied to Git commits.
- Reproducible pipelines that auto-detect what needs re-running.
- Cloud-agnostic remote storage with no vendor lock-in.
- Lightweight: only small metafiles are committed to Git.
- Built-in experiment tracking without additional services.
- Seamless collaboration through shared remote storage.

See `SKILL.md` for complete documentation and code examples.

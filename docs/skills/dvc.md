# DVC

The DVC skill covers data and model version control using DVC (Data Version Control), including remote storage, pipeline definitions, and experiment tracking for large binary assets.

**Skill directory:** `skills/dvc/`

## Purpose

Git is not designed for large binary files -- datasets, model weights, and video collections quickly bloat repositories. DVC provides Git-like version control for these assets, storing the actual files in remote storage (S3, GCS, Azure, SSH) while keeping lightweight pointer files in Git. This skill teaches Claude Code to configure DVC for CV/ML projects, define reproducible data pipelines, and manage dataset versions alongside code versions.

## When to Use

- Projects with datasets too large for Git (images, videos, point clouds)
- Model weight versioning across training runs
- Reproducible data processing pipelines (download -> preprocess -> train)
- Team workflows where multiple people need access to the same data versions

## Key Patterns

### DVC Initialization

```bash
# Initialize DVC in an existing Git repo
dvc init

# Configure remote storage
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc remote modify myremote region us-east-1
```

### Tracking Data

```bash
# Track a dataset directory
dvc add data/images/

# This creates data/images.dvc (pointer file) and adds data/images/ to .gitignore
git add data/images.dvc data/.gitignore
git commit -m "Track training images with DVC"

# Push data to remote
dvc push
```

### Pipeline Definition

```yaml
# dvc.yaml
stages:
  download:
    cmd: python scripts/download_data.py --output data/raw/
    outs:
      - data/raw/

  preprocess:
    cmd: python scripts/preprocess.py --input data/raw/ --output data/processed/
    deps:
      - data/raw/
      - scripts/preprocess.py
    outs:
      - data/processed/

  train:
    cmd: python src/my_project/train.py
    deps:
      - data/processed/
      - src/my_project/train.py
      - configs/
    outs:
      - models/best.ckpt
    metrics:
      - metrics.json:
          cache: false
```

### Reproduce Pipelines

```bash
# Run the full pipeline (only re-runs changed stages)
dvc repro

# Pull data from remote on a new machine
dvc pull
```

## Anti-Patterns to Avoid

- Do not commit large files to Git and then add DVC later -- initialize DVC at project start
- Do not use DVC for files under 10 MB -- Git handles those fine
- Avoid storing DVC cache on network-mounted filesystems -- it degrades performance
- Do not forget to `dvc push` after adding new data -- your teammates cannot `dvc pull` otherwise

## Combines Well With

- **PyTorch Lightning** -- Version model checkpoints alongside training code
- **Hydra Config** -- Pipeline parameters in DVC params files
- **GitHub Actions** -- DVC pull in CI for integration tests
- **Docker CV** -- Mount DVC-managed data into training containers

## Full Reference

See [`skills/dvc/SKILL.md`](https://github.com/ortizeg/ai-cv-claude-skills/blob/main/skills/dvc/SKILL.md) for patterns including DVC experiments, metrics comparison across branches, and integration with cloud storage providers.

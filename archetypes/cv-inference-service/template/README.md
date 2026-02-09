# ${project_name}

${description}

## Setup

```bash
uv sync --all-extras
```

## Running

```bash
# Development server
uv run uvicorn ${package_name}.app:app --reload --port 8000

# Production
uv run uvicorn ${package_name}.app:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

- `GET /health` — Health check
- `POST /api/v1/predict` — Run inference on an image

## Development

```bash
uv run pytest tests/ -v
uv run ruff check .
uv run mypy src/ --strict
```

FROM python:${python_version}-slim AS builder

WORKDIR /app
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM python:${python_version}-slim AS runtime

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$$PATH"

COPY src/ ./src/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "${package_name}.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

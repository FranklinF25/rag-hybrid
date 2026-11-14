# ---- build stage: resolve and install deps with uv ----
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
# Main dependencies only: no dev (pytest) / eval (deepeval) groups in runtime.
RUN uv sync --frozen --no-default-groups

# ---- runtime stage ----
FROM python:3.12-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src ./src
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
CMD ["rag-hybrid", "serve", "--host", "0.0.0.0"]

FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Build context is the project dir where pyproject.toml lives.
COPY . /app

RUN uv pip install --system --no-cache-dir -e .

ENV DISPATCH_ARENA_MAX_CONCURRENT_ENVS=16
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

EXPOSE 7860

CMD ["python", "-m", "dispatch_arena.server.app"]

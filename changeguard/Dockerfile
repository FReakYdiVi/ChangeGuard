FROM python:3.11-slim

WORKDIR /app

# Install uv (fast Python package manager).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Build context is the project dir (where pyproject.toml lives):
# docker build -f Dockerfile .
COPY . /app

# Install the changeguard-env package (server only — no training extras here).
RUN uv pip install --system --no-cache-dir -e .

ENV CHANGEGUARD_MAX_CONCURRENT_ENVS=16
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

EXPOSE 7860

CMD ["python", "-m", "changeguard.server.app"]

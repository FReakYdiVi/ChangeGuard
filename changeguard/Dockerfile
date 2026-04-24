FROM python:3.10-slim

WORKDIR /app

# Build context should be repo root when using:
# docker build -f changeguard/Dockerfile .
COPY . /app

RUN pip install --no-cache-dir --upgrade pip

ENV CHANGEGUARD_MAX_CONCURRENT_ENVS=16
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

EXPOSE 7860

CMD ["python", "-m", "changeguard.server.app"]

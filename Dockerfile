FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip

ENV CHANGEGUARD_MAX_CONCURRENT_ENVS=16
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

EXPOSE 7860

CMD ["python", "-c", "import os; from changeguard.server.app import run_local_server; run_local_server(host='0.0.0.0', port=int(os.getenv('PORT','7860')))"]

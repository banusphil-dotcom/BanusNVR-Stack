FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 curl ca-certificates \
        build-essential g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY server.py /app/server.py

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -fsS http://localhost:8765/v1/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8765"]

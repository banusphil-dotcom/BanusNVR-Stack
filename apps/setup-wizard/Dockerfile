FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi==0.115.* "uvicorn[standard]==0.32.*" \
    jinja2==3.1.* python-multipart==0.0.*

COPY server.py /app/server.py
COPY templates /app/templates
COPY static /app/static

EXPOSE 8090
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8090"]

FROM python:3.12-slim

WORKDIR /app

RUN pip install fastapi uvicorn[standard] httpx

COPY server.py /app/server.py
COPY registry.json /app/registry.json

EXPOSE 11434

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python3 -c "import httpx; httpx.get('http://localhost:11434/health', timeout=5.0).raise_for_status()" || exit 1

CMD ["python3", "/app/server.py"]

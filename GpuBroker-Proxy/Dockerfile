FROM python:3.12-slim@sha256:646fb0bca3dd3ea1bcc6feb72c17ed16eed6e10cffc732fcc1478bd3e7f02d7b

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r requirements.txt

COPY server.py /app/server.py
COPY registry.json /app/registry.json

EXPOSE 11434

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python3 -c "import httpx; httpx.get('http://localhost:11434/health', timeout=5.0).raise_for_status()" || exit 1

CMD ["python3", "/app/server.py"]

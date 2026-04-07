FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy everything
COPY . /app/

# Install only what the server needs (no openai, no httpx)
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn>=0.24.0" \
    "networkx>=3.0" \
    "pydantic>=2.0.0"

# PYTHONPATH so 'models' and 'server.*' resolve from /app
ENV PYTHONPATH="/app:$PYTHONPATH"

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

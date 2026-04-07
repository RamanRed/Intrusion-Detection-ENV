FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy project files into /app
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "uvicorn>=0.24.0" \
    "networkx>=3.0" \
    "pydantic>=2.0.0"

# Set PYTHONPATH so 'models', 'server.*' resolve correctly from /app
ENV PYTHONPATH="/app:$PYTHONPATH"

# Health check on port 7860 (HF Spaces default)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run the FastAPI server on port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

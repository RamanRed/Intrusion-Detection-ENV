FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy everything
COPY . /app/env

# Install Python dependencies directly from requirements.txt
RUN pip install --no-cache-dir \
    fastapi>=0.115.0 \
    uvicorn>=0.24.0 \
    networkx>=3.0 \
    pydantic>=2.0.0

# Set PYTHONPATH so 'models', 'server.*' resolve correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

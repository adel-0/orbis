FROM python:3.13-slim

WORKDIR /app

# Install system dependencies for CUDA and compilation
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy workspace files
COPY pyproject.toml .
COPY orbis-core orbis-core
COPY orbis orbis

WORKDIR /app/orbis

# Install PyTorch with CUDA support first
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cu129

# Install dependencies with ml extras (includes sentence-transformers)
# Also install orbis-core with ml extras
RUN uv pip install --system -e ../orbis-core --extra ml && \
    uv pip install --system -e . --extra ml

# Create data directories
RUN mkdir -p /app/orbis/data/database /app/orbis/data/chroma_db /app/orbis/data/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port
EXPOSE 7887

# Health check
HEALTHCHECK --interval=300s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7887/health || exit 1

# Use uvicorn directly for better control
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7887"]

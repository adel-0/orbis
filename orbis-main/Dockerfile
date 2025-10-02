FROM python:3.13-slim

WORKDIR /app

# Install system dependencies for CUDA and compilation
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install PyTorch with CUDA support first
RUN pip install torch --index-url https://download.pytorch.org/whl/cu129

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /app/data/database /app/data/chroma_db /app/data/models

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

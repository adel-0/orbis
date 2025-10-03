#!/bin/bash

# Orbis Docker Setup Script
# This script automates the Docker deployment process

set -e

echo "🚀 Orbis Docker Setup"
echo "================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker compose is available
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose v2 is not available. Please ensure Docker Desktop is installed or Docker Compose v2 is available."
    exit 1
fi

# Check for NVIDIA Docker (optional)
if ! docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi > /dev/null 2>&1; then
    echo "⚠️  NVIDIA Docker not detected. GPU acceleration may not be available."
    echo "   Install nvidia-docker2 for GPU support."
else
    echo "✅ NVIDIA Docker detected. GPU acceleration will be available."
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/database data/chroma_db data/models

# Check if .env exists
if [ ! -f .env ]; then
    if [ -f env.template ]; then
        echo "📝 Creating .env file from template..."
        cp env.template .env
        echo "⚠️  Please edit .env file with your configuration before continuing."
        echo "   At minimum, set AZURE_DEVOPS_ORG, AZURE_DEVOPS_PROJECT, AZURE_DEVOPS_PAT"
        read -p "Press Enter when you've configured .env..."
    else
        echo "❌ No .env file or env.template found. Please create .env manually."
        exit 1
    fi
else
    echo "✅ .env file found."
fi

# Build Docker image
echo "🔨 Building Docker image (this may take 10-15 minutes)..."
docker compose build

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
else
    echo "❌ Docker build failed. Please check the error messages above."
    exit 1
fi

# Start the service
echo "🚀 Starting Orbis API..."
docker compose up -d

# Wait for service to be ready
echo "⏳ Waiting for service to be ready..."
sleep 30

# Check health
echo "🏥 Checking service health..."
if curl -f http://localhost:7887/health > /dev/null 2>&1; then
    echo "✅ Service is healthy!"
    echo ""
    echo "🎉 Orbis is now running!"
    echo "   API: http://localhost:7887"
    echo "   Docs: http://localhost:7887/docs"
    echo "   Health: http://localhost:7887/health"
    echo ""
    echo "📊 Monitor logs with: docker compose logs -f oncall-api"
    echo "🛑 Stop service with: docker compose down"
else
    echo "❌ Service health check failed. Check logs with: docker compose logs oncall-api"
    exit 1
fi

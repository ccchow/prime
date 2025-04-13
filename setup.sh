#!/usr/bin/env bash
# Filename: setup.sh
# Description: Automate Prime environment setup and launch a 2-node (4 GPU total) training cluster.

set -e  # Exit immediately on any command failure
set -o pipefail

# 1. Clone the Prime repository from GitHub
echo "Cloning Prime repository..."
git clone https://github.com/PrimeIntellect-ai/prime.git || { echo "Git clone failed"; exit 1; }
cd prime

# 2. Build the Docker image for Prime (with a GPU-enabled base).
# Using PyTorch's CUDA-enabled base image for convenience. This image will include Python and CUDA libraries.
echo "Building Docker image 'prime:latest' for distributed training..."
cat > Dockerfile <<'EOF'
# Base image with CUDA, CUDNN and Python (PyTorch 2.1.0 with CUDA 11.8 for GPU support)
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# Install system dependencies (git for cloning submodules, others as needed)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git curl iperf && rm -rf /var/lib/apt/lists/*

# Add uv binary location to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Clone Prime repository into container and install it along with dependencies
WORKDIR /opt
RUN git clone --branch "task1" --recursive https://github.com/ccchow/prime.git \
 && cd prime \
 && curl -LsSf https://astral.sh/uv/install.sh | sh \
 && uv venv \
 && . .venv/bin/activate \
 && uv sync --extra all \
 && git submodule update --init --recursive

# Set working directory to Prime project
WORKDIR /opt/prime
EOF

# Build the Docker image (this may take a while as it installs dependencies and builds any extensions)
docker build -t prime:latest .

# 3. Launch the distributed training cluster using Docker Compose
# echo "Launching two Prime nodes with Docker Compose..."
# cd ..  # go back to project root where docker-compose.yml is expected
# docker compose up -d  || { echo "Docker Compose failed to start containers"; exit 1; }

# echo "Setup complete. The Prime training cluster is up and running."
# echo "Use 'docker compose logs -f node0' (or node1) to follow training logs, and 'docker compose ps' to see container status."


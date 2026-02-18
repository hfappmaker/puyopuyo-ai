FROM rust:latest

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Add NVIDIA CUDA repository (Debian 12 / Bookworm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg curl ca-certificates \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/3bf863cc.pub \
    | gpg --dearmor -o /usr/share/keyrings/cuda-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /" \
    > /etc/apt/sources.list.d/cuda.list \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA toolkit (nvcc, cublas, etc.)
# Note: allow-insecure needed because NVIDIA's repo key uses SHA1,
# which Debian trixie's sqv rejects since 2026-02-01
RUN apt-get update -o Acquire::AllowInsecureRepositories=true \
    && apt-get install -y --no-install-recommends --allow-unauthenticated \
    cuda-compiler-12-4 \
    libcublas-dev-12-4 \
    cuda-cudart-dev-12-4 \
    cuda-nvrtc-12-4 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/cuda-12.4/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64"

# Install wasm-pack and wasm target
RUN rustup target add wasm32-unknown-unknown \
    && cargo install wasm-pack

# Install Node.js 20 LTS and Git LFS
RUN curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
    | gpg --dearmor -o /usr/share/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
    > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update -o Acquire::AllowInsecureRepositories=true \
    && apt-get install -y --no-install-recommends --allow-unauthenticated nodejs git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Install Claude Code globally
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/root/.local/bin:${PATH}"

# NVIDIA runtime env
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace

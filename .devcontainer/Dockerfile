FROM rust:latest

# Install wasm-pack and wasm target
RUN rustup target add wasm32-unknown-unknown \
    && cargo install wasm-pack

# Install Node.js 20 LTS and Git LFS
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs git-lfs \
    && npm install -g npm@latest \
    && git lfs install

# Install Claude Code globally
RUN curl -fsSL https://claude.ai/install.sh | bash

ENV PATH="/root/.local/bin:${PATH}"
RUN echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

WORKDIR /workspace

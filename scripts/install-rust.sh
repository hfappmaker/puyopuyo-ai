#!/bin/bash
set -euo pipefail

# Rust インストール
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable

# PATH に追加
export PATH="/root/.cargo/bin:$PATH"

# CUDA パス設定
CUDA_PATH=$(ls -d /usr/local/cuda* 2>/dev/null | sort | tail -1)
if [ -n "$CUDA_PATH" ]; then
    export CUDA_PATH
    export CUDA_HOME="$CUDA_PATH"
    echo "CUDA_PATH=$CUDA_PATH"
fi

# 確認
rustc --version
cargo --version

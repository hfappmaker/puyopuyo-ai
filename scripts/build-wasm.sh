#!/bin/bash
set -e

echo "=== Running Rust tests ==="
cargo test --workspace

echo ""
echo "=== Building WASM ==="
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

wasm-pack build "$PROJECT_ROOT/crates/puyo/puyo-wasm" \
  --target web \
  --out-dir "$PROJECT_ROOT/web/wasm-pkg"

echo ""
echo "=== Installing web dependencies ==="
cd "$PROJECT_ROOT/web"
npm install

echo ""
echo "WASM build complete! Run 'cd web && npm run dev' to start."

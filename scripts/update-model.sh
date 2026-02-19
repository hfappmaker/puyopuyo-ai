#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SRC="$PROJECT_ROOT/artifacts/puyo_model_selfplay.bin"
DST_WEB="$PROJECT_ROOT/web/public/models/puyo_model_selfplay.bin"

if [ ! -f "$SRC" ]; then
    echo "Error: $SRC が見つかりません"
    exit 1
fi

# フロントエンド用に配置
cp "$SRC" "$DST_WEB"
echo "Copied: artifacts/puyo_model_selfplay.bin → web/public/models/puyo_model_selfplay.bin"

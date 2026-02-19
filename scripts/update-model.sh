#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SRC_MODEL="$PROJECT_ROOT/artifacts/puyo_model_selfplay.bin"
SRC_NORM="$PROJECT_ROOT/artifacts/norm_params.txt"
DST_DIR="$PROJECT_ROOT/web/public/models"

if [ ! -f "$SRC_MODEL" ]; then
    echo "Error: $SRC_MODEL が見つかりません"
    exit 1
fi
if [ ! -f "$SRC_NORM" ]; then
    echo "Error: $SRC_NORM が見つかりません"
    exit 1
fi

mkdir -p "$DST_DIR"

# フロントエンド用に配置
cp "$SRC_MODEL" "$DST_DIR/puyo_model_selfplay.bin"
echo "Copied: artifacts/puyo_model_selfplay.bin → web/public/models/puyo_model_selfplay.bin"
cp "$SRC_NORM" "$DST_DIR/norm_params.txt"
echo "Copied: artifacts/norm_params.txt → web/public/models/norm_params.txt"

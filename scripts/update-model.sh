#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

SRC_MODEL="$PROJECT_ROOT/artifacts/puyo_model.bin"
DST_DIR="$PROJECT_ROOT/web/public/models"

if [ ! -f "$SRC_MODEL" ]; then
    echo "Error: $SRC_MODEL が見つかりません"
    exit 1
fi

mkdir -p "$DST_DIR"

cp "$SRC_MODEL" "$DST_DIR/puyo_model.bin"
echo "Copied: artifacts/puyo_model.bin → web/public/models/puyo_model.bin"

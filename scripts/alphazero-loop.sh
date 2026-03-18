#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 設定（環境変数でオーバーライド可能）
GAMES="${GAMES:-100}"
SIMULATIONS="${SIMULATIONS:-25}"
C_PUCT="${C_PUCT:-1.5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
LOG_FILE="artifacts/alphazero-loop.log"

# イテレーションカウンタ（永続化）
ITER_FILE="artifacts/iteration.txt"
if [ -f "$ITER_FILE" ]; then
    ITERATION=$(cat "$ITER_FILE")
else
    ITERATION=1
fi

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

log "=== AlphaZero Loop Start (iteration=$ITERATION, games=$GAMES, sims=$SIMULATIONS) ==="

while true; do
    log "--- Iteration $ITERATION ---"
    SEED_OFFSET=$((ITERATION * GAMES))

    # 1. Self-play
    log "Self-play start (seed_offset=$SEED_OFFSET)"
    cargo run --release -p puyo-trainer --bin self-play -- \
        --games "$GAMES" \
        --simulations "$SIMULATIONS" \
        --c-puct "$C_PUCT" \
        --temperature "$TEMPERATURE" \
        --seed-offset "$SEED_OFFSET" \
        2>&1 | tee -a "$LOG_FILE"

    # 2. Git commit self-play data
    git add data/alphazero_data.bin
    git commit -m "alphazero: iter $ITERATION self-play (games=$GAMES, sims=$SIMULATIONS)"

    # 3. Train (GPU)
    log "Train start (AlphaZero mode)"
    cargo run --release -p puyo-trainer --bin train -- --alphazero \
        2>&1 | tee -a "$LOG_FILE"

    # 4. Git commit model
    git add artifacts/puyo_model.bin
    git commit -m "alphazero: iter $ITERATION training complete"

    # 5. 次のイテレーションへ
    ITERATION=$((ITERATION + 1))
    echo "$ITERATION" > "$ITER_FILE"

    log "=== Iteration $((ITERATION - 1)) complete ==="
done

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 設定（環境変数でオーバーライド可能）
GAMES="${GAMES:-100}"
SIMS_BASE="${SIMS_BASE:-25}"       # シミュレーション初期値
SIMS_STEP="${SIMS_STEP:-5}"        # イテレーションごとの増加量
SIMS_MAX="${SIMS_MAX:-200}"        # シミュレーション上限
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

log "=== AlphaZero Loop Start (iteration=$ITERATION, games=$GAMES, sims=$SIMS_BASE+$SIMS_STEP/iter, max=$SIMS_MAX) ==="

while true; do
    # シミュレーション数: SIMS_BASE + (ITERATION - 1) * SIMS_STEP（上限 SIMS_MAX）
    SIMS=$((SIMS_BASE + (ITERATION - 1) * SIMS_STEP))
    if [ "$SIMS" -gt "$SIMS_MAX" ]; then
        SIMS=$SIMS_MAX
    fi

    log "--- Iteration $ITERATION (sims=$SIMS) ---"
    SEED_OFFSET=$((ITERATION * GAMES))

    # 1. Self-play
    log "Self-play start (seed_offset=$SEED_OFFSET)"
    cargo run --release -p puyo-trainer --bin self-play -- \
        --games "$GAMES" \
        --simulations "$SIMS" \
        --c-puct "$C_PUCT" \
        --temperature "$TEMPERATURE" \
        --seed-offset "$SEED_OFFSET" \
        2>&1 | tee -a "$LOG_FILE"

    # 2. Git commit self-play data
    git add data/alphazero_data.bin
    git commit -m "alphazero: iter $ITERATION self-play (games=$GAMES, sims=$SIMS)"

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

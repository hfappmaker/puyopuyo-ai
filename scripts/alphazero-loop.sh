#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 設定（環境変数でオーバーライド可能）
GAMES="${GAMES:-300}"
SIMS_BASE="${SIMS_BASE:-200}"       # シミュレーション初期値
SIMS_STEP="${SIMS_STEP:-0}"        # イテレーションごとの増加量
SIMS_MAX="${SIMS_MAX:-200}"        # シミュレーション上限
C_PUCT="${C_PUCT:-1.5}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DIRICHLET_ALPHA="${DIRICHLET_ALPHA:-0.4}"
DIRICHLET_EPSILON="${DIRICHLET_EPSILON:-0.25}"
TEMP_THRESHOLD="${TEMP_THRESHOLD:-15}"
REPLAY_WINDOW="${REPLAY_WINDOW:-3}"  # 直近N個のイテレーションデータを保持
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

    OUTPUT_FILE="data/alphazero_iter_${ITERATION}.bin"

    # 1. Self-play
    log "Self-play start (seed_offset=$SEED_OFFSET, dirichlet_alpha=$DIRICHLET_ALPHA)"
    cargo run --release -p puyo-trainer --bin self-play -- \
        --games "$GAMES" \
        --simulations "$SIMS" \
        --c-puct "$C_PUCT" \
        --temperature "$TEMPERATURE" \
        --seed-offset "$SEED_OFFSET" \
        --dirichlet-alpha "$DIRICHLET_ALPHA" \
        --dirichlet-epsilon "$DIRICHLET_EPSILON" \
        --temp-threshold "$TEMP_THRESHOLD" \
        --output "$OUTPUT_FILE" \
        2>&1 | tee -a "$LOG_FILE"

    # 2. Remove old data files beyond replay window
    OLD_ITER=$((ITERATION - REPLAY_WINDOW))
    if [ "$OLD_ITER" -gt 0 ]; then
        OLD_FILE="data/alphazero_iter_${OLD_ITER}.bin"
        if [ -f "$OLD_FILE" ]; then
            log "Removing old data: $OLD_FILE"
            rm -f "$OLD_FILE"
        fi
    fi

    # 3. Git commit self-play data
    git add data/alphazero_iter_*.bin
    git add -u data/  # stage deletions
    git commit -m "alphazero: iter $ITERATION self-play (games=$GAMES, sims=$SIMS)" || true

    # 4. Train (GPU) with replay buffer
    log "Train start (AlphaZero mode, replay buffer)"
    cargo run --release -p puyo-trainer --bin train -- --alphazero --data-dir data \
        2>&1 | tee -a "$LOG_FILE"

    # 5. Git commit model
    git add artifacts/puyo_model.bin
    git commit -m "alphazero: iter $ITERATION training complete"

    # 5. 次のイテレーションへ
    ITERATION=$((ITERATION + 1))
    echo "$ITERATION" > "$ITER_FILE"

    log "=== Iteration $((ITERATION - 1)) complete ==="
done

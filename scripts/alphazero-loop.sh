#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 設定（環境変数でオーバーライド可能）
GAMES="${GAMES:-300}"
SIMS_BASE="${SIMS_BASE:-64}"        # シミュレーション初期値
SIMS_STEP="${SIMS_STEP:-0}"        # イテレーションごとの増加量
SIMS_MAX="${SIMS_MAX:-64}"         # シミュレーション上限
C_PUCT="${C_PUCT:-1.5}"
M="${M:-16}"                        # Gumbel Top-k初期サンプル数
C_VISIT="${C_VISIT:-50.0}"            # Q値スケーリング
GAMMA="${GAMMA:-0.95}"              # 割引率
REPLAY_WINDOW="${REPLAY_WINDOW:-30}"  # 直近N個のイテレーションデータを保持
MIN_CHAIN="${MIN_CHAIN:-0}"          # 最低連鎖数フィルタ（0=無効）
LOG_FILE="artifacts/alphazero-loop.log"

# イテレーションカウンタ（永続化）
ITER_FILE="artifacts/iteration.txt"
if [ -f "$ITER_FILE" ]; then
    ITERATION=$(cat "$ITER_FILE")
else
    ITERATION=1
    mkdir -p artifacts
    echo "$ITERATION" > "$ITER_FILE"
fi

# グローバルステップカウンタ（永続化、LRスケジュール用）
GLOBAL_STEP_FILE="artifacts/global_step.txt"
if [ -f "$GLOBAL_STEP_FILE" ]; then
    GLOBAL_STEP=$(cat "$GLOBAL_STEP_FILE")
else
    GLOBAL_STEP=0
    echo "$GLOBAL_STEP" > "$GLOBAL_STEP_FILE"
fi

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

log "=== AlphaZero Loop Start (iteration=$ITERATION, games=$GAMES, sims=$SIMS_BASE+$SIMS_STEP/iter, max=$SIMS_MAX, min_chain=$MIN_CHAIN) ==="

while true; do
    # シミュレーション数: SIMS_BASE + (ITERATION - 1) * SIMS_STEP（上限 SIMS_MAX）
    SIMS=$((SIMS_BASE + (ITERATION - 1) * SIMS_STEP))
    if [ "$SIMS" -gt "$SIMS_MAX" ]; then
        SIMS=$SIMS_MAX
    fi

    log "--- Iteration $ITERATION (sims=$SIMS, global_step=$GLOBAL_STEP) ---"

    OUTPUT_FILE="data/alphazero_iter_${ITERATION}.bin"

    # 1. Self-play (Gumbel MCTS)
    log "Self-play start (m=$M, c_visit=$C_VISIT)"
    cargo run --release -p puyo-trainer --bin self-play -- \
        --games "$GAMES" \
        --simulations "$SIMS" \
        --c-puct-init "$C_PUCT" \
        --m "$M" \
        --c-visit "$C_VISIT" \
        --gamma "$GAMMA" \
        --min-chain "$MIN_CHAIN" \
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
    log "Train start (AlphaZero mode, replay buffer, global_step=$GLOBAL_STEP)"
    cargo run --release -p puyo-trainer --bin train -- --alphazero --data-dir data \
        --global-step "$GLOBAL_STEP" \
        2>&1 | tee -a "$LOG_FILE"

    # Update global step from training output
    if [ -f "$GLOBAL_STEP_FILE" ]; then
        GLOBAL_STEP=$(cat "$GLOBAL_STEP_FILE")
    fi

    # 5. Git commit model
    git add artifacts/puyo_model.bin artifacts/iteration.txt artifacts/global_step.txt
    git commit -m "alphazero: iter $ITERATION training complete"

    # 6. 次のイテレーションへ
    ITERATION=$((ITERATION + 1))
    echo "$ITERATION" > "$ITER_FILE"

    log "=== Iteration $((ITERATION - 1)) complete ==="
done

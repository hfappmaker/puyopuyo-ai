#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ブランチ名（必須引数）
if [ $# -lt 1 ]; then
    echo "Usage: $0 <branch-name>"
    echo "Example: $0 train/run-001"
    exit 1
fi
TARGET_BRANCH="$1"

# ブランチ名からランディレクトリを作成（/ を - に置換）
RUN_DIR="runs/$(echo "$TARGET_BRANCH" | tr '/' '-')"
ARTIFACTS_DIR="$RUN_DIR/artifacts"
DATA_DIR="$RUN_DIR/data"
MODEL_PATH="$ARTIFACTS_DIR/puyo_model"
mkdir -p "$ARTIFACTS_DIR" "$DATA_DIR"

# 指定ブランチに切り替え（なければ作成、リモートのみの場合は追跡ブランチを作成）
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "$TARGET_BRANCH" ]; then
    git checkout "$TARGET_BRANCH" 2>/dev/null || git checkout -b "$TARGET_BRANCH"
fi

# 設定（環境変数でオーバーライド可能）
GAMES="${GAMES:-300}"
SIMS_BASE="${SIMS_BASE:-64}"        # シミュレーション初期値
SIMS_STEP="${SIMS_STEP:-0}"        # イテレーションごとの増加量
SIMS_MAX="${SIMS_MAX:-64}"         # シミュレ���ション上限
C_PUCT_INIT="${C_PUCT_INIT:-1.5}"
C_PUCT_BASE="${C_PUCT_BASE:-19652.0}"  # PUCT探索ベース定数
M="${M:-16}"                        # Gumbel Top-k初期サンプル数
C_VISIT="${C_VISIT:-50.0}"            # Q値スケーリング
GAMMA="${GAMMA:-0.95}"              # 割引率
REPLAY_WINDOW="${REPLAY_WINDOW:-30}"  # 直近N個のイテレーションデータを保持
MIN_CHAIN="${MIN_CHAIN:-0}"          # 最低連鎖数フィルタ（0=無効）
THREADS="${THREADS:-128}"            # self-playスレッド数
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-512}"  # GPU推論バッチサイズ
NUM_LEAVES="${NUM_LEAVES:-24}"                # MCTS virtual loss バッチ（同時探索リーフ数）
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"  # 学習バッチサイズ
ACCUM_STEPS="${ACCUM_STEPS:-4}"              # 勾配累積ステップ数
BOARD_COLS="${BOARD_COLS:-3}"                # ボード列数
BOARD_ROWS="${BOARD_ROWS:-8}"               # ボード行数
NUM_COLORS="${NUM_COLORS:-3}"               # 色数
RESIDUAL_CHANNELS="${RESIDUAL_CHANNELS:-64}" # モデルチャネル数
NUM_BLOCKS="${NUM_BLOCKS:-6}"               # モデルブロック数
POLICY_CONV_CHANNELS="${POLICY_CONV_CHANNELS:-2}"   # Policy head Conv チャネル数
VALUE_CONV_CHANNELS="${VALUE_CONV_CHANNELS:-1}"     # Value head Conv チャネル数
VALUE_HIDDEN="${VALUE_HIDDEN:-64}"                   # Value head FC 隠れ層サイズ
FILM_HIDDEN="${FILM_HIDDEN:-128}"                    # FiLM 隠れ層サイズ
# ログファイル名を算出（100イテレーションごとにローテーション）
update_log_file() {
    local start=$(( ((ITERATION - 1) / 100) * 100 + 1 ))
    local end=$(( start + 99 ))
    LOG_FILE="$ARTIFACTS_DIR/alphazero-loop-$(printf '%04d' $start)-$(printf '%04d' $end).log"
}

# イテレーションカウンタ（永続化）
ITER_FILE="$ARTIFACTS_DIR/iteration.txt"
if [ -f "$ITER_FILE" ]; then
    ITERATION=$(cat "$ITER_FILE")
else
    ITERATION=1
    echo "$ITERATION" > "$ITER_FILE"
fi

# グローバルステップカウンタ（永続化、LRスケジュール用）
GLOBAL_STEP_FILE="$ARTIFACTS_DIR/global_step.txt"
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

commit_and_push() {
    local msg="$1"
    shift
    # 引数で渡されたファイル/パターンをステージング
    for pattern in "$@"; do
        git add $pattern 2>/dev/null || true
    done
    git add -u "$DATA_DIR/"  # stage deletions
    if git diff --cached --quiet; then
        log "No changes to commit"
        return
    fi
    git commit -m "$msg" || true
    git push -u origin "$TARGET_BRANCH" || log "WARNING: push failed, will retry next commit"
}

update_log_file
log "=== AlphaZero Loop Start (branch=$TARGET_BRANCH, run_dir=$RUN_DIR, iteration=$ITERATION, games=$GAMES, sims=$SIMS_BASE+$SIMS_STEP/iter, max=$SIMS_MAX, min_chain=$MIN_CHAIN, threads=$THREADS) ==="

while true; do
    # シミュレーション数: SIMS_BASE + (ITERATION - 1) * SIMS_STEP（上限 SIMS_MAX���
    SIMS=$((SIMS_BASE + (ITERATION - 1) * SIMS_STEP))
    if [ "$SIMS" -gt "$SIMS_MAX" ]; then
        SIMS=$SIMS_MAX
    fi

    update_log_file
    log "--- Iteration $ITERATION (sims=$SIMS, global_step=$GLOBAL_STEP) ---"

    OUTPUT_FILE="$DATA_DIR/alphazero_iter_${ITERATION}.bin"

    # 1. Self-play (Gumbel MCTS)
    log "Self-play start (m=$M, c_visit=$C_VISIT)"
    cargo run --release -p puyo-trainer --bin self-play -- \
        --games "$GAMES" \
        --simulations "$SIMS" \
        --c-puct-init "$C_PUCT_INIT" \
        --c-puct-base "$C_PUCT_BASE" \
        --m "$M" \
        --c-visit "$C_VISIT" \
        --gamma "$GAMMA" \
        --min-chain "$MIN_CHAIN" \
        --threads "$THREADS" \
        --batch-size "$INFER_BATCH_SIZE" \
        --num-leaves "$NUM_LEAVES" \
        --model-path "$MODEL_PATH" \
        --cols "$BOARD_COLS" \
        --rows "$BOARD_ROWS" \
        --num-colors "$NUM_COLORS" \
        --residual-channels "$RESIDUAL_CHANNELS" \
        --num-blocks "$NUM_BLOCKS" \
        --policy-conv-channels "$POLICY_CONV_CHANNELS" \
        --value-conv-channels "$VALUE_CONV_CHANNELS" \
        --value-hidden "$VALUE_HIDDEN" \
        --film-hidden "$FILM_HIDDEN" \
        --output "$OUTPUT_FILE" \
        2>&1 | tee -a "$LOG_FILE"

    # 2. Remove old data files beyond replay window
    OLD_ITER=$((ITERATION - REPLAY_WINDOW))
    if [ "$OLD_ITER" -gt 0 ]; then
        OLD_FILE="$DATA_DIR/alphazero_iter_${OLD_ITER}.bin"
        if [ -f "$OLD_FILE" ]; then
            log "Removing old data: $OLD_FILE"
            rm -f "$OLD_FILE"
        fi
    fi

    # 3. Commit & push self-play data
    commit_and_push "alphazero: iter $ITERATION self-play (games=$GAMES, sims=$SIMS)" \
        "$DATA_DIR/alphazero_iter_*.bin" "$ARTIFACTS_DIR/alphazero-loop-*.log"

    # 4. Train (GPU) with replay buffer
    log "Train start (AlphaZero mode, replay buffer, global_step=$GLOBAL_STEP)"
    cargo run --release -p puyo-trainer --bin train -- --alphazero \
        --data-dir "$DATA_DIR" \
        --global-step "$GLOBAL_STEP" \
        --model-path "$MODEL_PATH" \
        --artifacts-dir "$ARTIFACTS_DIR" \
        --batch-size "$TRAIN_BATCH_SIZE" \
        --accum-steps "$ACCUM_STEPS" \
        --cols "$BOARD_COLS" \
        --rows "$BOARD_ROWS" \
        --num-colors "$NUM_COLORS" \
        --residual-channels "$RESIDUAL_CHANNELS" \
        --num-blocks "$NUM_BLOCKS" \
        --policy-conv-channels "$POLICY_CONV_CHANNELS" \
        --value-conv-channels "$VALUE_CONV_CHANNELS" \
        --value-hidden "$VALUE_HIDDEN" \
        --film-hidden "$FILM_HIDDEN" \
        2>&1 | tee -a "$LOG_FILE"

    # Update global step from training output
    if [ -f "$GLOBAL_STEP_FILE" ]; then
        GLOBAL_STEP=$(cat "$GLOBAL_STEP_FILE")
    fi

    # 5. 次のイテレーションへ
    ITERATION=$((ITERATION + 1))
    echo "$ITERATION" > "$ITER_FILE"

    # 6. Commit & push model
    commit_and_push "alphazero: iter $((ITERATION - 1)) training complete" \
        "$ARTIFACTS_DIR/puyo_model.bin" "$ARTIFACTS_DIR/puyo_model.config.json" "$ITER_FILE" "$GLOBAL_STEP_FILE" "$ARTIFACTS_DIR/alphazero-loop-*.log"

    log "=== Iteration $((ITERATION - 1)) complete ==="
done

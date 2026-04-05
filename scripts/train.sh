# 使い方
# source scripts/train.sh
# bash scripts/alphazero-loop-push.sh develop

# --- ゲーム設定 ---
export BOARD_COLS=3
export BOARD_ROWS=8
export NUM_COLORS=3

# --- モデル設定 ---
export RESIDUAL_CHANNELS=64
export NUM_BLOCKS=6
export POLICY_CONV_CHANNELS=2
export VALUE_CONV_CHANNELS=1
export VALUE_HIDDEN=64
export FILM_HIDDEN=128

# --- 学習パラメータ ---
export GAMES=300
export SIMS_BASE=64
export SIMS_STEP=0
export SIMS_MAX=64
export C_PUCT_INIT=1.5
export C_PUCT_BASE=19652.0
export M=16
export C_VISIT=50.0
export GAMMA=0.95
export REPLAY_WINDOW=30
export MIN_CHAIN=0
export THREADS=128
export INFER_BATCH_SIZE=128
export NUM_LEAVES=1
export TRAIN_BATCH_SIZE=512
export ACCUM_STEPS=4

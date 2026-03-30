//! ゲームパラメータの一元管理モジュール。
//!
//! ボードサイズ・色数・消去条件などの定数を変更する場合は、このファイルだけを編集して
//! 再コンパイルしてください。NN モデルの再学習も必要です。

// ─── 基本パラメータ ───

/// ボードの列数（横幅）。
pub const COLS: usize = 6;

/// ボードの行数（表示行 + 隠し2行）。
pub const ROWS: usize = 14;

/// 表示行数。上2行は隠し行でゲームオーバー判定に使う。
pub const VISIBLE_ROWS: usize = ROWS - 2;

/// ぷよの出現列（0-indexed、ボード中央）。
pub const SPAWN_COL: usize = (COLS - 1) / 2;

/// 使用する色の数（1〜4）。PuyoColor の先頭 N 色を使う。
pub const NUM_COLORS: usize = 4;

/// 連鎖で消えるために必要な同色ぷよの最小接続数。
pub const MIN_GROUP_SIZE: usize = 4;

// ─── 派生定数（NN テンソル関連） ───

/// 入力チャンネル数: 色ごとの one-hot + occupancy + adjacency。
pub const NUM_CHANNELS: usize = NUM_COLORS + 2;

/// 盤面テンソルのフラットサイズ。
pub const TENSOR_SIZE: usize = NUM_CHANNELS * ROWS * COLS;

/// ピースエンコーディングのサイズ: 3ピース × 2色 × NUM_COLORS one-hot。
pub const PIECE_TENSOR_SIZE: usize = 3 * 2 * NUM_COLORS;

/// コンテキストテンソルサイズ。
pub const CONTEXT_TENSOR_SIZE: usize = PIECE_TENSOR_SIZE;

/// アクション空間のサイズ: COLS × 4方向。
pub const NUM_ACTIONS: usize = COLS * 4;

// ─── コンパイル時バリデーション ───

const _: () = assert!(NUM_COLORS >= 1 && NUM_COLORS <= 4, "NUM_COLORS must be 1..=4");
const _: () = assert!(COLS >= 1, "COLS must be >= 1");
const _: () = assert!(ROWS == VISIBLE_ROWS + 2, "ROWS must be VISIBLE_ROWS + 2");
const _: () = assert!(MIN_GROUP_SIZE >= 2, "MIN_GROUP_SIZE must be >= 2");

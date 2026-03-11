# WASM ブリッジ仕様

## 概要

Rust で実装したゲームロジックとAIを、`wasm-bindgen` を使ってブラウザから利用できるようにする橋渡し層。

## WasmGame クラス

`#[wasm_bindgen]` で公開される主要クラス。内部に `GameState` を保持する。

## API 一覧

### ゲーム管理

| メソッド | 引数 | 戻り値 | 説明 |
|---------|------|--------|------|
| `new(seed)` | `u64` | `WasmGame` | シードを指定してゲームを初期化 |
| `restart(seed)` | `u64` | - | 新しいシードでゲームをリセット |

### 状態取得

| メソッド | 戻り値 | 説明 |
|---------|--------|------|
| `get_board()` | `Vec<u8>` (長さ84) | 盤面データ。列優先・下から上。6列×14行。各バイトは PuyoColor (0-4) |
| `get_current_piece()` | `Vec<u8>` (長さ6 or 0) | `[axis_color, sat_color, col, row_int, row_frac×100, orientation]` |
| `get_next_piece()` | `Vec<u8>` (長さ2) | `[axis_color, sat_color]` |
| `get_next_next_piece()` | `Vec<u8>` (長さ2) | `[axis_color, sat_color]` |
| `get_score()` | `u32` | 現在のスコア |
| `get_max_chain()` | `u32` | 最大連鎖数 |
| `get_phase()` | `u8` | ゲームフェーズ。`GamePhase::as_u8()` で変換（0=Falling, 1=Resolving, 2=GameOver） |
| `get_total_pieces()` | `u32` | 設置済みぷよ組数 |

### 操作

| メソッド | 戻り値 | 説明 |
|---------|--------|------|
| `move_left()` | `bool` | 左移動。成功なら `true` |
| `move_right()` | `bool` | 右移動。成功なら `true` |
| `rotate_cw()` | `bool` | 時計回り回転。成功なら `true` |
| `rotate_ccw()` | `bool` | 反時計回り回転。成功なら `true` |
| `hard_drop()` | `u32` | ハードドロップ。発生した連鎖数を返す |
| `soft_drop()` | `bool` | 1マス下降。移動できたら `true`、着地位置なら `false` |
| `tick(gravity)` | `u32` | 重力による落下。着地して連鎖が発生したら連鎖数を返す |

### AI

| メソッド | 戻り値 | 説明 |
|---------|--------|------|
| `ai_best_move()` | `Vec<u8>` (長さ2 or 0) | `[col, orientation]` 形式で最善手を返す。orientation は `Orientation::as_u8()` で変換 |
| `ai_play_move()` | `u32` | 最善手を計算し即座に適用。発生した連鎖数を返す |

### モデル管理

| メソッド | 引数 | 戻り値 | 説明 |
|---------|------|--------|------|
| `load_nn_model(model_bytes, mean, std_dev)` | `&[u8]`, `f32`, `f32` | - | NN モデルをバイト列から読み込み、評価器を `NnEvaluator` に切り替える |
| `use_heuristic()` | - | - | 評価器を `HeuristicEvaluator` に切り替える |

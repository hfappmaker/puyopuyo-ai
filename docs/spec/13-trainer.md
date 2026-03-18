# 訓練パイプライン仕様

## 概要

SimulationEvaluator AI の対戦データを元に CNN Dual Head Network を教師あり学習で訓練し、さらに MCTS ベースの AlphaZero self-play で強化学習するパイプライン。

## クレート構成

```
crates/puyo-trainer/
├── src/
│   ├── lib.rs                  # モジュール公開
│   ├── data.rs                 # 訓練データ構造
│   └── bin/
│       ├── generate_data.rs    # データ生成バイナリ
│       ├── train.rs            # 教師あり学習バイナリ
│       └── self_play.rs        # 自己対戦強化学習バイナリ
└── Cargo.toml
```

## 依存関係

| クレート | 用途 |
|---------|------|
| `puyo-core` | ゲームロジック |
| `puyo-ai` | シミュレーション評価・探索 |
| `puyo-nn` | CNN モデル・エンコーディング |
| `burn` | NN フレームワーク（ndarray, autodiff, train） |
| `serde` / `bincode` | データのシリアライズ |

## 訓練データ (`data`)

### Sample

```rust
pub struct Sample {
    pub board_data: Vec<f32>,    // エンコード済み盤面 (504 floats)
    pub context_data: Vec<f32>,  // コンテキストエンコーディング (24 floats: 3ツモ × 2色 × 4 one-hot)
    pub action_index: u8,        // SimulationEvaluator が選択した配置インデックス (0〜23)
}
```

### Dataset

```rust
pub struct Dataset {
    pub samples: Vec<Sample>,
}
```

### AlphaZeroSample

MCTS self-play で生成されるサンプル。Policy と Value の両方の教師信号を含む。

```rust
pub struct AlphaZeroSample {
    pub board_data: Vec<f32>,     // エンコード済み盤面 (504 floats)
    pub context_data: Vec<f32>,   // コンテキストエンコーディング (24 floats)
    pub mcts_policy: Vec<f32>,    // MCTS 探索による配置確率分布 (24 floats)
    pub value_target: f32,        // 累積割引報酬（γ=0.99 で逆算）
}
```

### AlphaZeroDataset

```rust
pub struct AlphaZeroDataset {
    pub samples: Vec<AlphaZeroSample>,
}
```

保存形式: `bincode` によるバイナリシリアライズ。

## Phase 1: データ生成 (`generate-data`)

SimulationEvaluator AI に自動対戦させ、訓練データを収集する。

### パラメータ

| 名前 | 値 | 説明 |
|------|-----|------|
| `NUM_GAMES` | 10,000 | 対戦回数 |
| `OUTPUT_PATH` | `data/training_data.bin` | 出力先 |
| `MAX_MOVES_PER_GAME` | 50 | 1ゲームあたりの最大手数 |

### 手順

1. シード `0..NUM_GAMES` で各ゲームを実行
2. 各手番で盤面状態を `board_to_tensor_data` で記録
3. 3ツモを `context_to_tensor_data` で記録
4. `find_best_move`（`SimulationEvaluator`）で最善手を選択
5. 選択された配置を `placement_to_index()` でインデックスに変換して記録
6. 配置を適用
7. 最大手数（`MAX_MOVES_PER_GAME`）に達するかゲーム終了まで繰り返す

## Phase 2: 教師あり学習 (`train`)

生成データで Dual Head Network（`PuyoNet`）を学習する。`--alphazero` フラグで AlphaZero モード（Policy CE + Value MSE）と教師ありモード（Policy CE のみ）を切り替える。バックエンドは `NdArray`（CPU）または `CudaJit`（GPU、`gpu` feature flag）+ `Autodiff`。

### パラメータ

| 名前 | 値 | 説明 |
|------|-----|------|
| `BATCH_SIZE` | 512 (GPU) / 64 (CPU) | バッチサイズ |
| `NUM_EPOCHS` | 50 | 最大エポック数 |
| `LR_MAX` | 5e-4 | Cosine Annealing 初期学習率 |
| `LR_MIN` | 1e-5 | Cosine Annealing 最終学習率 |
| `EARLY_STOPPING_PATIENCE` | 5 | Early Stopping の patience（エポック数） |
| `MODEL_PATH` | `artifacts/puyo_model` | モデル保存先 |

### 学習率スケジューラ（Cosine Annealing）

エポック単位で学習率を Cosine 減衰させる。

```
lr(epoch) = LR_MIN + 0.5 × (LR_MAX - LR_MIN) × (1 + cos(π × epoch / NUM_EPOCHS))
```

### Early Stopping

Validation loss が `EARLY_STOPPING_PATIENCE` エポック連続で改善しない場合、学習を早期終了する。Validation loss が改善するたびにベストモデルを保存する。

### 手順

1. データを `TRAIN_SPLIT_RATIO`（0.9）で訓練/検証に分割
2. エポックごとに Cosine Annealing で学習率を計算
3. LCG（PCG family パラメータ: `6364136223846793005`, `1`）ベースのシャッフル → ミニバッチ学習
4. 損失関数: Cross-Entropy（action_index を正解ラベルとして使用）
5. 最適化: Adam
6. Validation loss 改善時にベストモデルを `BinFileRecorder` で保存
7. Early Stopping 判定（patience=5）

## Phase 3: 自己対戦強化学習 (`self-play`)

MCTS ベースの AlphaZero self-play ループ。Dual Head Network（`PuyoNet`）の Policy Head と Value Head を使った MCTS 探索でゲームをプレイし、訓練データを生成する。

### 手順

1. 現在のモデルを使って MCTS 探索でゲームをプレイ
2. 各手番で MCTS の訪問回数分布を policy target として記録
3. ゲーム終了後、各手番の value target を累積割引報酬で逆算:
   ```
   value_target[t] = Σ_{k=0}^{T-t-1} γ^k × score[t+k]  （γ = 0.99）
   ```
4. `AlphaZeroSample`（board_data, context_data, mcts_policy, value_target）を生成し、`data/alphazero_data.bin` に保存
5. 生成データは `train --alphazero` で Policy Head（Cross-Entropy 損失）と Value Head（MSE 損失）を同時に学習

## 実行順序

```bash
cargo run --bin generate-data            # Phase 1: データ生成
cargo run --bin train                    # Phase 2: 教師あり学習（Policy CE のみ）
cargo run --bin self-play                # Phase 3: 自己対戦データ生成
cargo run --bin train -- --alphazero     # Phase 3: AlphaZero 学習（Policy CE + Value MSE）
```

## 成果物

| ファイル | 説明 |
|---------|------|
| `data/training_data.bin` | 教師あり学習データ（bincode） |
| `data/alphazero_data.bin` | AlphaZero self-play データ（bincode） |
| `artifacts/puyo_model` | 学習済みモデル（教師あり / AlphaZero 共通保存先） |

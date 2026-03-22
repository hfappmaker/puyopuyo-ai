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
    pub context_data: Vec<f32>,  // コンテキストエンコーディング (24 floats: ツモ one-hot)
    pub action_index: u8,        // SimulationEvaluator が選択した配置インデックス (0〜23)
    pub value_target: f32,       // 教師評価器によるスコア推定値
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
    pub context_data: Vec<f32>,   // コンテキストエンコーディング (24 floats: ツモ one-hot)
    pub mcts_policy: Vec<f32>,    // MCTS 探索による配置確率分布 (24 floats)
    pub value_target: f32,        // 累積割引報酬（γ=0.95 で逆算）
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

生成データで Dual Head Network（`PuyoNet`）を学習する。`--alphazero` フラグで AlphaZero モード（Policy CE + Value MSE）と教師ありモード（Policy CE のみ）を切り替える。AlphaZero モードでは value_target に MuZero Invertible Value Transform（`value_transform()`）を適用してから MSE 損失を計算する（詳細は `docs/spec/12-nn.md` の Value Transform セクション参照）。バックエンドは `NdArray`（CPU）または `CudaJit`（GPU、`gpu` feature flag）+ `Autodiff`。

### パラメータ

| 名前 | 値 | 説明 |
|------|-----|------|
| `BATCH_SIZE` | 512 (GPU) / 64 (CPU) | バッチサイズ |
| `NUM_EPOCHS` | 50（教師あり） / 40（AlphaZero） | 最大エポック数 |
| `LR_MAX` | 5e-4（教師あり） / 2e-4（AlphaZero） | Cosine Annealing 初期学習率 |
| `LR_MIN` | 1e-5 | Cosine Annealing 最終学習率 |
| `EARLY_STOPPING_PATIENCE` | 5（教師あり） / 10（AlphaZero） | Early Stopping の patience（エポック数） |
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
4. 損失関数: Policy は Cross-Entropy（action_index を正解ラベルとして使用）
5. 最適化: Adam（Weight decay 1e-4 付き）
6. Validation loss 改善時にベストモデルを `BinFileRecorder` で保存
7. Early Stopping 判定（patience=5）

#### Weight Decay

Adam に Weight decay（L2正則化）を追加している:

```rust
AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1e-4))).init()
```

過学習を抑制し、汎化性能を向上させる目的で、教師あり学習・AlphaZero 学習の両モードに適用される。

## Phase 3: 自己対戦強化学習 (`self-play`)

Gumbel MCTS ベースの AlphaZero self-play ループ。Dual Head Network（`PuyoNet`）の Policy Head と Value Head を使った Gumbel MCTS 探索でゲームをプレイし、訓練データを生成する。
各ゲームは `std::thread::scope` により並列実行される（スレッド数 = CPU コア数）。各スレッドがモデルのクローンを所有し、独立にゲームを処理する。

### CLI引数

| 引数 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `--games` | 整数 | 100 | 自己対戦ゲーム数 |
| `--simulations` | 整数 | 64 | MCTS シミュレーション回数/手 |
| `--c-puct` | 小数 | 1.5 | 内部ノードのPUCT探索定数 |
| `--seed-offset` | 整数 | 200,000 | RNG シードオフセット（seed = seed_offset + game_idx） |
| `--m` | 整数 | 16 | Gumbel Top-k 初期サンプル数 |
| `--c-visit` | 小数 | 50.0 | Q値スケーリング係数 |
| `--c-scale` | 小数 | 1.0 | Advantage スケールパラメータ |
| `--gamma` | 小数 | 0.95 | 将来報酬の割引率 |
| `--output` | 文字列 | `data/alphazero_data.bin` | 出力ファイルパス |

`--seed-offset` により、複数回の self-play 実行で異なるゲームデータを生成できる。

### 探索の多様性

Gumbel AlphaZero では、各手番でGumbel(0,1)ノイズをサンプリングすることで探索の多様性を確保する。Dirichletノイズや温度スケジュールは不要。Gumbelノイズのシードはゲームシードと手数から決定論的に生成される。

### 手順

1. 現在のモデルを使って Gumbel MCTS 探索でゲームをプレイ（`mcts_search()` に `gamma` を渡す）
2. 各手番で improved policy（completed Q-values に基づく改善ポリシー）を policy target として記録
3. ゲーム終了後、各手番の value target を累積割引報酬で逆算:
   ```
   value_target[t] = Σ_{k=0}^{T-t-1} γ^k × score[t+k]  （γ = 0.95）
   ```
4. **MAX_TURNS 打ち切り時はブートストラップ**：ゲームオーバーではなく手数上限で打ち切られた場合、NN の value 推定を使って最終ターンの value target を補正する:
   ```
   value_target[last] = reward[last] + GAMMA × V_nn(final_state)
   ```
   `estimate_value()` が `value_inverse_transform()` を適用した実スケール推定値を返す。これにより、手数打ち切りによる value の過小評価（未来スコアの切り捨て）を緩和し、学習品質を改善する
5. `AlphaZeroSample`（board_data, context_data, mcts_policy, value_target）を生成し、出力ファイル（デフォルト: `data/alphazero_data.bin`、`--output` で変更可能）に保存
6. 生成データは `train --alphazero` で Policy Head（Cross-Entropy 損失）と Value Head（MSE 損失、MuZero Invertible Value Transform 適用、重み `VALUE_LOSS_WEIGHT=0.5`）を同時に学習

#### AlphaZero 学習のデータシャッフル

AlphaZero モードでは、train/val 分割の**前に**全データをシャッフルしてから分割する。

自己対局データはゲーム単位で連続して格納されるため、シャッフルなしで分割すると訓練/検証データがゲームの前半/後半に偏る（系統的バイアス）。事前シャッフルにより、各分割セットにゲームの多様な局面が均一に含まれる。

### リプレイバッファ

`alphazero-loop.sh` では、各イテレーションのデータを `data/alphazero_iter_N.bin` として個別に保存する。`train --alphazero --data-dir data` で直近のイテレーションデータを全て結合して学習に使用する（リプレイバッファ）。

- `REPLAY_WINDOW`（デフォルト10）で保持するイテレーション数を制御
- 古いデータは自動削除される
- `AlphaZeroDataset::load_multiple()` で複数ファイルをマージ

## 実行順序

```bash
cargo run --bin generate-data            # Phase 1: データ生成
cargo run --bin train                    # Phase 2: 教師あり学習（Policy CE のみ）
cargo run --bin self-play                # Phase 3: 自己対戦データ生成
cargo run --bin train -- --alphazero     # Phase 3: AlphaZero 学習（Policy CE + Value MSE）
```

## 自動化ループ (`scripts/alphazero-loop.sh`)

self-play → git commit → train --alphazero → git commit を無限ループで繰り返すスクリプト。

```bash
# デフォルト設定で実行
bash scripts/alphazero-loop.sh

# パラメータをオーバーライド
GAMES=200 SIMULATIONS=50 bash scripts/alphazero-loop.sh
```

- イテレーション番号は `artifacts/iteration.txt` に永続化（中断・再開に対応）
- ログは `artifacts/alphazero-loop.log` に記録
- `--seed-offset` は `iteration * games` で自動計算（イテレーションごとに異なるデータ）

## 成果物

| ファイル | 説明 |
|---------|------|
| `data/training_data.bin` | 教師あり学習データ（bincode） |
| `data/alphazero_data.bin` | AlphaZero self-play データ（レガシー単一ファイル、bincode） |
| `data/alphazero_iter_N.bin` | イテレーション別 self-play データ（リプレイバッファ用） |
| `artifacts/puyo_model` | 学習済みモデル（教師あり / AlphaZero 共通保存先） |

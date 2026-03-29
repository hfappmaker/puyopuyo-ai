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
    pub board_data: Vec<f32>,    // エンコード済み盤面 (TENSOR_SIZE floats = 120)
    pub context_data: Vec<f32>,  // コンテキストエンコーディング (CONTEXT_TENSOR_SIZE floats = 18: ツモ one-hot)
    pub action_index: u8,        // SimulationEvaluator が選択した配置インデックス (0〜NUM_ACTIONS-1)
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
    pub board_data: Vec<f32>,     // エンコード済み盤面 (TENSOR_SIZE floats = 120)
    pub context_data: Vec<f32>,   // コンテキストエンコーディング (CONTEXT_TENSOR_SIZE floats = 18: ツモ one-hot)
    pub mcts_policy: Vec<f32>,    // MCTS 探索による配置確率分布 (NUM_ACTIONS floats = 12)
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

生成データで Dual Head Network（`PuyoNet`）を学習する。`--alphazero` フラグで AlphaZero モード（Soft Policy CE + Value MSE）と教師ありモード（Hard Policy CE + Value MSE）を切り替える。AlphaZero モードでは value_target に MuZero Invertible Value Transform（`value_transform()`）を適用してから MSE 損失を計算する（詳細は `docs/spec/12-nn.md` の Value Transform セクション参照）。バックエンドは `NdArray`（CPU）または `CudaJit`（GPU、`gpu` feature flag）+ `Autodiff`。

### パラメータ

| 名前 | 値 | 説明 |
|------|-----|------|
| `BATCH_SIZE` | 512 (GPU) / 64 (CPU) | バッチサイズ |
| `NUM_EPOCHS` | 50 | 教師あり学習の最大エポック数 |
| `LR_MAX` / `LR_MIN` | 5e-4 / 1e-5 | 教師あり学習の Cosine Annealing 学習率範囲 |
| `AZ_NUM_STEPS` | 1,000 | AlphaZero 学習のステップ数（エポックではなくステップベース） |
| `AZ_LR_MAX` / `AZ_LR_MIN` | 2e-4 / 1e-5 | AlphaZero 学習の Cosine Annealing 学習率範囲 |
| `EARLY_STOPPING_PATIENCE` | 5 | 教師あり学習の Early Stopping patience（AlphaZero モードでは不使用） |
| `TRAIN_SPLIT_RATIO` | 0.9 | 教師あり学習の訓練/検証データ分割比率 |
| `VALUE_LOSS_WEIGHT` | 0.5 | Value 損失の重み係数 |
| `MODEL_PATH` | `artifacts/puyo_model` | モデル保存先 |

### 学習率スケジューラ（Cosine Annealing）

教師あり学習ではエポック単位、AlphaZero 学習ではステップ単位で学習率を Cosine 減衰させる。

```
# 教師あり学習
lr(epoch) = LR_MIN + 0.5 × (LR_MAX - LR_MIN) × (1 + cos(π × epoch / NUM_EPOCHS))

# AlphaZero 学習
lr(step) = AZ_LR_MIN + 0.5 × (AZ_LR_MAX - AZ_LR_MIN) × (1 + cos(π × step / AZ_NUM_STEPS))
```

### Early Stopping（教師あり学習のみ）

Validation loss が `EARLY_STOPPING_PATIENCE` エポック連続で改善しない場合、学習を早期終了する。Validation loss が改善するたびにベストモデルを保存する。AlphaZero モードでは Early Stopping を使用せず、全ステップ完了後にモデルを保存する（性能は self-play の報酬で判断）。

### 教師あり学習の手順

1. データを `TRAIN_SPLIT_RATIO`（0.9）で訓練/検証に分割
2. エポックごとに Cosine Annealing で学習率を計算
3. LCG（PCG family パラメータ: `6364136223846793005`, `1`）ベースのシャッフル → ミニバッチ学習
4. 損失関数: Policy は Cross-Entropy（action_index を正解ラベルとして使用）+ Value は MSE（`value_transform` 適用）
5. 最適化: Adam（Weight decay 1e-4 付き）
6. Validation loss 改善時にベストモデルを `BinFileRecorder` で保存
7. Early Stopping 判定（patience=5）

### AlphaZero 学習の手順

1. 全データを訓練に使用（val split なし — 性能は self-play の報酬で判断）
2. ステップごとに Cosine Annealing で学習率を計算（`AZ_LR_MAX` → `AZ_LR_MIN`）
3. 各ステップでランダムミニバッチをサンプリングし、オンザフライで色置換データ拡張を適用
4. 損失関数: Policy は Soft Cross-Entropy（MCTS 訪問分布を正解ラベル、無効アクションをマスク）+ Value は MSE（`value_transform` 適用、重み `VALUE_LOSS_WEIGHT=0.5`）
5. 最適化: Adam（Weight decay 1e-4 付き）
6. 全ステップ完了後にモデルを `BinFileRecorder` で保存
7. 既存モデルがあれば読み込んで継続学習（なければランダム初期化）

#### Weight Decay

Adam に Weight decay（L2正則化）を追加している:

```rust
AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1e-4))).init()
```

過学習を抑制し、汎化性能を向上させる目的で、教師あり学習・AlphaZero 学習の両モードに適用される。

## Phase 3: 自己対戦強化学習 (`self-play`)

Gumbel MCTS ベースの AlphaZero self-play ループ。Dual Head Network（`PuyoNet`）の Policy Head と Value Head を使った Gumbel MCTS 探索でゲームをプレイし、訓練データを生成する。
CPU モードでは `std::thread::scope` により並列実行され、各スレッドがモデルのクローンを所有して独立にゲームを処理する。GPU モードでは専用の推論サーバースレッド（`inference_server`）がバッチ推論を処理し、ゲームスレッド（デフォルト128）が `InferenceClient` 経由で推論リクエストを送信する。`--threads` 引数でスレッド数を指定可能。

### CLI引数

| 引数 | 型 | デフォルト | 説明 |
|------|-----|----------|------|
| `--games` | 整数 | 300 | 自己対戦ゲーム数 |
| `--simulations` | 整数 | 64 | MCTS シミュレーション回数/手 |
| `--c-puct` | 小数 | 1.5 | 内部ノードのPUCT探索定数 |
| `--seed-offset` | 整数 | 200,000 | RNG シードオフセット（seed = seed_offset + game_idx） |
| `--m` | 整数 | 16 | Gumbel Top-k 初期サンプル数 |
| `--c-visit` | 小数 | 5.0 | Q値スケーリング係数 |
| `--gamma` | 小数 | 0.95 | 将来報酬の割引率 |
| `--output` | 文字列 | `data/alphazero_data.bin` | 出力ファイルパス |
| `--threads` | 整数 | CPUコア数（GPU: 128） | 並列ゲームスレッド数 |
| `--batch-size` | 整数 | 128（GPU のみ） | GPU 推論サーバーの最大バッチサイズ |
| `--min-chain` | 整数 | 0（無効） | 最低連鎖数フィルタ。指定値未満のmax_chainのゲームを除外 |

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

#### AlphaZero 学習のデータサンプリング

AlphaZero モードでは val split を行わず、全データを訓練に使用する。各ステップでランダムにミニバッチをサンプリングする（LCG ベースの乱数でインデックスを選択）。エポック単位のシャッフルではなくステップ単位のランダムサンプリングにより、データの偏りを回避する。

### 色置換データ拡張（Color Permutation Augmentation）

学習時に各サンプルに対してランダムな色置換を適用し、データを実質24倍に拡張する。ぷよぷよでは4色の入れ替えはゲームの意味を変えないため、等価な訓練データを生成できる。

- **適用タイミング**: AlphaZero 学習時の各ステップで、ミニバッチサンプリングと同時にオンザフライで適用（ランダムに1置換を選択）
- **置換数**: NUM_COLORS! 通り（NUM_COLORS=3 なら 3! = 6通り、恒等置換を含む）
- **ColorPermutation**: `Vec<usize>`（NUM_COLORS に応じた可変長）。`all_color_permutations()` が `Vec<ColorPermutation>` を返す
- **PLANE_SIZE**: ROWS × COLS (= 24)
- **適用対象**: `board_data`（ch0〜(NUM_COLORS-1) の色one-hotチャンネルを入れ替え）と `context_data`（6つの NUM_COLORS 要素 one-hot ブロックを入れ替え）。`apply_color_perm_board` / `apply_color_perm_context` は NUM_COLORS でパラメータ化されている
- **不変項目**: `mcts_policy`（アクションは列×方向で色に依存しない）、`value_target`（累積スコア）、`board_data` の ch NUM_COLORS（占有）・ch NUM_COLORS+1（隣接度）

### リプレイバッファ

`alphazero-loop.sh` では、各イテレーションのデータを `data/alphazero_iter_N.bin` として個別に保存する。`train --alphazero --data-dir data` で直近のイテレーションデータを全て結合して学習に使用する（リプレイバッファ）。

- `REPLAY_WINDOW`（デフォルト10）で保持するイテレーション数を制御
- 古いデータは自動削除される
- `AlphaZeroDataset::load_multiple()` で複数ファイルをマージ

## 実行順序

```bash
cargo run --bin generate-data            # Phase 1: データ生成
cargo run --bin train                    # Phase 2: 教師あり学習（Policy CE + Value MSE）
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

# 5連鎖以上のゲームのみ学習データとして保存
MIN_CHAIN=5 bash scripts/alphazero-loop.sh
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

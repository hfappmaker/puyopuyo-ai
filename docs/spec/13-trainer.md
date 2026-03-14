# 訓練パイプライン仕様

## 概要

ヒューリスティック AI の対戦データを元に CNN 価値ネットワークを訓練し、さらに自己対戦で強化するパイプライン。

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
| `puyo-ai` | ヒューリスティック評価・探索 |
| `puyo-nn` | CNN モデル・エンコーディング |
| `burn` | NN フレームワーク（ndarray, autodiff, train） |
| `serde` / `bincode` | データのシリアライズ |

## 訓練データ (`data`)

### Sample

```rust
pub struct Sample {
    pub board_data: Vec<f32>,  // エンコード済み盤面 (840 floats)
    pub target: f32,           // 目標値: 割引済み将来スコア
}
```

### Dataset

```rust
pub struct Dataset {
    pub samples: Vec<Sample>,
}
```

保存形式: `bincode` によるバイナリシリアライズ。

## Phase 1: データ生成 (`generate-data`)

ヒューリスティック AI に自動対戦させ、訓練データを収集する。

### パラメータ

| 名前 | 値 | 説明 |
|------|-----|------|
| `NUM_GAMES` | 10,000 | 対戦回数 |
| `OUTPUT_PATH` | `data/training_data.bin` | 出力先 |
| `MAX_MOVES_PER_GAME` | 50 | 1ゲームあたりの最大手数 |

### 手順

1. シード `0..NUM_GAMES` で各ゲームを実行
2. 各手番で盤面状態を `board_to_tensor_data` で記録
3. `find_best_move`（`SimulationEvaluator`）で最善手を選択・適用
4. 設置ごとのスコアを記録
5. 最大手数（`MAX_MOVES_PER_GAME`）に達するかゲーム終了まで繰り返す
6. ゲーム終了後、割引累積報酬（γ = 0.95）を逆方向に計算

### 目標値の計算

```
future_values[最終手] = score[最終手]
future_values[t] = score[t] + γ × future_values[t+1]
```

## Phase 2: 教師あり学習 (`train`)

生成データで CNN を学習する。バックエンドは `NdArray` + `Autodiff`。

### パラメータ

| 名前 | 値 | 説明 |
|------|-----|------|
| `BATCH_SIZE` | 512 | バッチサイズ |
| `NUM_EPOCHS` | 20 | エポック数 |
| `LEARNING_RATE` | 5e-4 | 学習率 |
| `MODEL_PATH` | `artifacts/puyo_model` | モデル保存先 |

### 手順

1. データを `TRAIN_SPLIT_RATIO`（0.9）で訓練/検証に分割
2. 訓練セットの目標値を標準化（平均0、標準偏差1）
3. 正規化パラメータ（mean, std_dev）を `artifacts/norm_params.txt` に保存
4. エポックごとに LCG（PCG family パラメータ: `6364136223846793005`, `1`）ベースのシャッフル → ミニバッチ学習
5. 損失関数: MSE
6. 最適化: Adam
7. 学習済みモデルを `BinFileRecorder` で保存

## Phase 3: 自己対戦強化学習 (`self-play`)

学習済みモデルを評価関数として探索に使用し、TD(λ) のバッファ方式で学習する。1配置 = 1遷移として扱い、連鎖解決は `apply_placement()` で一括実行する。ゲームオーバー後は盤面をリセットして即座に続行する。

### パラメータ

| 名前 | 値 | 説明 |
|------|-----|------|
| `TOTAL_STEPS` | 200,000 | 全体のステップ数（遷移数ベース） |
| `GAMMA` | 0.99 | 割引率 |
| `LAMBDA` | 0.8 | TD(λ) の λ パラメータ |
| `BUFFER_SIZE` | 64 | 軌跡バッファの容量 |
| `LEARNING_RATE` | 1e-4 | 学習率 |
| `EPSILON_START` | 0.3 | 初期探索率 |
| `EPSILON_END` | 0.01 | 最終探索率 |
| `TARGET_UPDATE_INTERVAL` | 1,000 | ターゲットネットワーク更新間隔（ステップ数） |
| `LOG_INTERVAL` | 100 | 進捗ログ出力間隔（ステップ数） |

### コマンドラインオプション

| オプション | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| `--steps` | u64 | 200,000 | 全体のステップ数 |
| `--target-update` | u64 | 1,000 | ターゲット更新間隔 |
| `--buffer-size` | usize | 64 | バッファ容量 |
| `--lambda` | f32 | 0.8 | λ パラメータ |

### コード構造

`self_play.rs` は以下の構造体・関数で構成される。

| 名前 | 種別 | 役割 |
|------|------|------|
| `NormParams` | 構造体 | z-score 正規化パラメータ。`normalize()`, `eval_target()` を提供 |
| `RewardStats` | 構造体 | 報酬カウンタと損失累積。`record(reward, loss)`, `log_and_reset()` でログ管理 |
| `GameStats` | 構造体 | ゲームパフォーマンス統計（連鎖数・手数の移動平均）。収束確認用 |
| `GameSession` | 構造体 | ゲーム状態（`GameState`, seed, カウンタ）。`reset()`, `log_game_over_and_reset()` |
| `Transition` | 構造体 | TD(λ) バッファの1遷移（`board_data`, `reward`, `terminal`） |
| `TrajectoryBuffer` | 構造体 | 遷移バッファ。`push()`, `is_full()`, `clear()` を提供 |
| `TrainingContext<O>` | 構造体 | 学習に関する可変状態を一括管理。`optim`, `buffer`, `target_model`, `step`, `stats`, `game_stats` 等を保持。`flush_buffer()` と `batch_update()` をメソッドとして提供 |
| `SelfPlayEvaluator` | 構造体 | `Evaluator` トレイト実装。ターゲットモデルで盤面を評価 |
| `compute_lambda_returns()` | 関数 | バッファを後ろから走査してλ-returnを計算 |
| `sync_target_network()` | 関数 | ターゲットネットワークをオンラインモデルから同期 |
| `select_placement()` | 関数 | ε-greedy 配置選択。`Option<Placement>` を返す |
| `compute_epsilon()` | 関数 | εの線形減衰計算 |
| `simple_rng()` | 関数 | splitmix64 アルゴリズムによる決定論的 RNG（ε-greedy 用） |
| `run_training_loop()` | 関数 | メインの学習ループ |

### TrainingContext

`TrainingContext<O>` は学習ループ内の可変状態を集約する構造体で、`&mut self` 以外の `&mut` パラメータを排除している。

```rust
struct TrainingContext<O> {
    optim: O,
    buffer: TrajectoryBuffer,
    target_model: PuyoValueNet<InferBackend>,
    step: u64,
    stats: RewardStats,
    game_stats: GameStats,
    config: PuyoValueNetConfig,
    norm: NormParams,
    // ... デバイス・ハイパーパラメータ
}
```

主要メソッド:
- `batch_update(&mut self, model, targets) -> (model, loss)`: バッファの全遷移をまとめて1回のforward + backwardで学習する。model を消費して更新済み model と平均損失を返す
- `flush_buffer(&mut self, model, session, epsilon) -> model`: λ-return計算 → バッチ学習 → 統計記録 → ターゲット更新判定 → バッファクリアを一括実行。model を消費して更新済み model を返す

### 探索率（ε）

ステップ数に基づく線形減衰。`EPSILON_START (0.3)` から `EPSILON_END (0.01)` へ全ステップにわたって線形に減少する。

```
ε = EPSILON_START + (EPSILON_END - EPSILON_START) × (step / TOTAL_STEPS)
```

### 報酬関数

1配置 = 1遷移として、連鎖解決は `apply_placement()` で一括実行する。

| 条件 | 報酬 |
|------|------|
| 連鎖が発生 | スコア（`chain_result.score`） |
| 連鎖なし（生存） | 0 |
| ゲームオーバー | -1 |

### 遷移バッファと TD(λ)

各配置で1つの `Transition` をバッファに追加する。バッファが満杯（`BUFFER_SIZE` 遷移）またはゲームオーバー時に `TrainingContext::flush_buffer()` を呼び、バッファ内の全遷移をまとめて学習する。

#### λ-return の計算（`compute_lambda_returns()`）

バッファを末尾から走査し、各遷移のλ-returnを計算する:

```
g = if 末尾がterminal { 0.0 } else { V(last_next_board) }
for t in (0..n).rev():
    if transitions[t].terminal:
        g = transitions[t].reward  // -1、伝搬を断ち切る
    else:
        v_next = if t+1 < n { V(transitions[t+1].board_data) } else { V(last_next_board) }
        g = reward[t] + γ × ((1-λ) × v_next + λ × g)
    targets[t] = normalize(g)
```

#### バッチ学習（`TrainingContext::batch_update()`）

バッファの全盤面データを `[n, NUM_CHANNELS, ROWS, COLS]` テンソルにまとめ、1回の forward + backward で MSE 損失を計算・逆伝播する。`TrainingContext` のメソッドとして実装されている。

### エピソードレス設計

ゲームオーバーは terminal=true の遷移として扱い、λ-return の伝搬を断ち切る。ゲームオーバー後は新しいシードで `GameState::new()` を呼び、盤面をリセットして即座にプレイを続行する。ゲームオーバー時はバッファを即座に消化する。

### 手順

1. Phase 2 で学習したモデル（`artifacts/puyo_model`）と正規化パラメータをロード
2. ターゲットネットワーク（凍結コピー）を用意
3. 各ステップで ε-greedy 方策を使用:
   - 確率 ε: ランダム配置
   - 確率 1-ε: ターゲットネットワーク評価 + 2手先読み探索で最善手を選択
4. `game.apply_placement()` でピース配置 + 連鎖解決を一括実行
5. 遷移をバッファに追加:
   - **ゲームオーバー**: `reward=-1, terminal=true` → バッファを即座に消化
   - **生存**: `reward=連鎖数, terminal=false` → バッファが満杯なら消化
6. `TrainingContext::flush_buffer()` で TD(λ) 学習:
   - `compute_lambda_returns()` でλ-returnを計算
   - `TrainingContext::batch_update()` でバッチ学習（MSE損失）
7. `TARGET_UPDATE_INTERVAL` ステップごとにターゲットネットワークを現在のモデルで更新（一時ファイル経由）
8. 最終モデルを `artifacts/puyo_model_selfplay` に保存

### ログ出力と収束指標

`LOG_INTERVAL`（100ステップ）ごとに以下の収束指標をログ出力する。

```
[PROGRESS] step=100/200000, games=5, eps=0.299, rewards(-1/0/+1)=10/20/70, loss=0.0342, avg_chain=4.2, avg_moves=18.5
```

| フィールド | 説明 | 収束時の傾向 |
|-----------|------|------------|
| `loss` | バッチMSE損失（`LOG_INTERVAL`ステップ平均） | 減少または安定 |
| `avg_chain` | 直近ゲームの最大連鎖数の平均 | 増加 |
| `avg_moves` | 直近ゲームの手数の平均 | 増加 |
| `rewards(-1/0/+1)` | 報酬分布（ゲームオーバー/連鎖なし/連鎖あり） | 正報酬が増加 |

`TrainingContext::flush_buffer()` 内で各遷移の報酬を `RewardStats` に記録し、`batch_update()` の損失値を累積する。`GameStats` はゲーム終了ごとに連鎖数と手数を記録し、ログ時に平均を計算してリセットする。

### NN 評価関数

`SelfPlayEvaluator` は `puyo-ai` の `Evaluator` トレイトを実装し、探索エンジンに組み込まれる。`NormParams` への参照を保持し、非正規化を委譲する。`find_best_move` は `NnEvaluator` と同じ BFS 統一パターン（depth-1/2/3 の全盤面を評価し、最高スコアの1手目を返す）を採用している。

```rust
struct SelfPlayEvaluator<'a> {
    model: &'a PuyoValueNet<InferBackend>,
    device: <InferBackend as Backend>::Device,
    norm: &'a NormParams,
}
```

## 実行順序

```bash
cargo run --bin generate-data   # Phase 1: データ生成
cargo run --bin train            # Phase 2: 教師あり学習
cargo run --bin self-play        # Phase 3: 自己対戦強化学習
```

## 成果物

| ファイル | 説明 |
|---------|------|
| `data/training_data.bin` | 訓練データ（bincode） |
| `artifacts/norm_params.txt` | 正規化パラメータ（mean, std_dev） |
| `artifacts/puyo_model` | 教師あり学習済みモデル |
| `artifacts/puyo_model_selfplay` | 自己対戦強化学習済みモデル |

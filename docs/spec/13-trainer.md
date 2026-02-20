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
    pub board_data: Vec<f32>,  // one-hot 盤面 (390 floats)
    pub target: f32,           // 目標値: 割引済み将来連鎖数
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

### 手順

1. シード `0..NUM_GAMES` で各ゲームを実行
2. 各手番で盤面状態を `board_to_tensor_data` で記録
3. `find_best_move`（ヒューリスティック評価）で最善手を選択・適用
4. 設置ごとの連鎖数を記録
5. ゲーム終了後、割引累積報酬（γ = 0.95）を逆方向に計算

### 目標値の計算

```
future_values[最終手] = chain_count[最終手]
future_values[t] = chain_count[t] + γ × future_values[t+1]
```

## Phase 2: 教師あり学習 (`train`)

生成データで CNN を学習する。バックエンドは `NdArray` + `Autodiff`。

### パラメータ

| 名前 | 値 | 説明 |
|------|-----|------|
| `BATCH_SIZE` | 512 | バッチサイズ |
| `NUM_EPOCHS` | 20 | エポック数 |
| `LEARNING_RATE` | 1e-3 | 学習率 |
| `MODEL_PATH` | `artifacts/puyo_model` | モデル保存先 |

### 手順

1. データを 90:10 で訓練/検証に分割
2. 訓練セットの目標値を標準化（平均0、標準偏差1）
3. 正規化パラメータ（mean, std_dev）を `artifacts/norm_params.txt` に保存
4. エポックごとに xorshift128+ RNG でシャッフル → ミニバッチ学習
5. 損失関数: MSE
6. 最適化: Adam
7. 学習済みモデルを `CompactRecorder` で保存

## Phase 3: 自己対戦強化学習 (`self-play`)

学習済みモデルを評価関数として探索に使用し、TD(0) で毎ステップ更新する。エピソードの概念はなく、ゲームオーバー後は盤面をリセットして即座に続行する。

### パラメータ

| 名前 | 値 | 説明 |
|------|-----|------|
| `TOTAL_STEPS` | 200,000 | 全体のステップ数（TD更新回数ベース） |
| `GAMMA` | 0.99 | 割引率 |
| `LEARNING_RATE` | 1e-4 | 学習率 |
| `EPSILON_START` | 0.3 | 初期探索率 |
| `EPSILON_END` | 0.01 | 最終探索率 |
| `TARGET_UPDATE_INTERVAL` | 1,000 | ターゲットネットワーク更新間隔（ステップ数） |
| `LOG_INTERVAL` | 1,000 | 進捗ログ出力間隔（ステップ数） |

### 探索率（ε）

ステップ数に基づく線形減衰。`EPSILON_START (0.3)` から `EPSILON_END (0.01)` へ全ステップにわたって線形に減少する。

```
ε = EPSILON_START + (EPSILON_END - EPSILON_START) × (step / TOTAL_STEPS)
```

### 報酬関数

配置と連鎖解決を分離し、連鎖の各ステップに報酬を与える。

| 条件 | 報酬 |
|------|------|
| 配置して連鎖が発生（配置ステップ） | 0 |
| 連鎖の各ステップ（ぷよが消えるたび） | +1 |
| 配置して連鎖なし（生存） | +1 |
| ゲームオーバー | -1 |

#### 連鎖時のステップ分解

連鎖が発生した場合、`chain::resolve_one_step()` で1連鎖ずつ盤面を進め、各ステップでTD更新を行う。

例: 3連鎖の場合 → 4回のTD更新

| # | state | reward | next_state |
|---|-------|--------|------------|
| 1 | 配置前盤面 | 0 | 配置後盤面（消去前） |
| 2 | 配置後盤面（消去前） | +1 | 1連鎖消去後盤面 |
| 3 | 1連鎖消去後盤面 | +1 | 2連鎖消去後盤面 |
| 4 | 2連鎖消去後盤面 | +1 | 3連鎖消去後盤面 |

連鎖なしの場合 → 1回のTD更新: state=配置前盤面, reward=+1, next_state=配置後盤面。
ゲームオーバーの場合 → 1回のTD更新: state=配置前盤面, reward=-1, next_state=リセット後盤面（生存報酬は与えない）。

`TOTAL_STEPS` は配置回数ではなくTD更新回数を数える。連鎖が多いほど1配置で複数ステップを消費する。

### エピソードレス設計

ゲームオーバーはゲーム終了ではなく、-1 の報酬が発生するイベントとして扱う。ゲームオーバー後は新しいシードで `GameState::new()` を呼び、盤面をリセットして即座にプレイを続行する。`V(next_state)` はリセット後の新しい盤面で計算する。

### 手順

1. Phase 2 で学習したモデル（`artifacts/puyo_model`）と正規化パラメータをロード
2. ターゲットネットワーク（凍結コピー）を用意
3. 各ステップで ε-greedy 方策を使用:
   - 確率 ε: ランダム配置
   - 確率 1-ε: ターゲットネットワーク評価 + 2手先読み探索で最善手を選択
4. `game.place_piece_only()` でピースを配置（連鎖は解決しない）
5. 連鎖の有無を `chain::find_groups()` で判定:
   - **連鎖あり**: 配置前→配置後で reward=0 のTD更新、`chain::resolve_one_step()` で1連鎖ずつ解決しながら reward=+1 のTD更新。連鎖完了後に `finalize_after_chains()` でゲームオーバー判定し、ゲームオーバーなら追加の reward=-1 TD更新
   - **連鎖なし**: `finalize_after_chains()` で先にゲームオーバー判定。ゲームオーバーなら reward=-1 のTD更新のみ、そうでなければ reward=+1（生存報酬）のTD更新
6. TD(0) 更新式:
   ```
   V(next_state) = target_model(next_board) * std_dev + mean  # 非正規化
   td_target = (reward + γ × V(next_state) - mean) / std_dev  # 正規化
   loss = MSE(V(state), td_target)
   ```
7. `TARGET_UPDATE_INTERVAL` ステップごとにターゲットネットワークを現在のモデルで更新（一時ファイル経由）
8. 最終モデルを `artifacts/puyo_model_selfplay` に保存

### NN 評価関数

`SelfPlayEvaluator` は `puyo-ai` の `Evaluator` トレイトを実装し、探索エンジンに組み込まれる。

```rust
impl Evaluator for SelfPlayEvaluator {
    fn evaluate(&self, board: &Board) -> f64 {
        // ゲームオーバー → -100000.0
        // それ以外: NN の出力を非正規化して返す
    }
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

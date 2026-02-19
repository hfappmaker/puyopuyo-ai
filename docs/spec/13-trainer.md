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

学習済みモデルを評価関数として探索に使用し、Monte Carlo 法で更新する。

### パラメータ

| 名前 | 値 | 説明 |
|------|-----|------|
| `NUM_GAMES` | 5,000 | 対戦回数 |
| `GAMMA` | 0.99 | 割引率 |
| `LEARNING_RATE` | 1e-4 | 学習率 |
| `EPSILON_HIGH` | 0.3 | カリキュラム序盤（フェーズ0）の探索率 |
| `EPSILON_START` | 0.1 | フェーズ1以降の初期探索率 |
| `EPSILON_END` | 0.01 | 最終探索率 |
| `CHAIN_WINDOW` | 20 | 連鎖数移動平均のウィンドウサイズ（ゲーム数） |
| `TARGET_UPDATE_INTERVAL` | 20 | ターゲットネットワーク更新間隔（ゲーム数） |
| `MAX_MOVES_PER_GAME` | 50 | 1ゲームあたりの最大手数 |
| `GAME_OVER_PENALTY` | -100.0 | ゲームオーバー時の終端ペナルティ |

### カリキュラム学習

直近 `CHAIN_WINDOW` ゲームの連鎖数移動平均 `avg_chain` に基づき、学習に使うゲームの品質閾値を段階的に引き上げる。

昇格条件は `promote_at = min_chain + 1.0 / min_chain` で動的に計算する。フェーズが上がるほど閾値が `min_chain` に近づき、緩やかな昇格設計になる。

| フェーズ | 学習する最小連鎖数 | 昇格条件（avg_chain ≥） |
|---------|------------------|----------------------|
| 0 | 1 | 2.0 (1 + 1/1) |
| 1 | 2 | 2.5 (2 + 1/2) |
| 2 | 3 | 3.33 (3 + 1/3) |
| 3 | 4 | 4.25 (4 + 1/4) |
| 4 | 5 | 5.2 (5 + 1/5) |
| 5 | 6 | 6.17 (6 + 1/6) |
| 6 | 7 | 7.14 (7 + 1/7) |
| 7 | 8 | 8.125 (8 + 1/8) |
| 8（最終） | 9 | - |

- フェーズ0: `ε = EPSILON_HIGH (0.3)`（探索重視）
- フェーズ1以降: 初回昇格ゲームを起点に `ε` を線形減衰（0.1 → 0.01）

### 報酬関数

各ステップの即時報酬は連鎖数の二乗とする。

```
reward = chain_count²
```

### 手順

1. Phase 2 で学習したモデル（`artifacts/puyo_model`）と正規化パラメータをロード
2. ターゲットネットワーク（凍結コピー）を用意
3. 各ゲームで ε-greedy 方策を使用:
   - 確率 ε: ランダム配置
   - 確率 1-ε: ターゲットネットワーク評価 + 2手先読み探索で最善手を選択
4. ゲームが `MAX_MOVES_PER_GAME` 手に達するか、ゲームオーバーになるまで繰り返す
5. ゲーム終了後、Monte Carlo 法で割引リターンを計算（逆方向）:
   ```
   terminal_bonus = GAME_OVER_PENALTY  (ゲームオーバー時) or 0.0
   running_return = terminal_bonus
   for t in (T-1)..0:
       running_return = reward[t] + γ × running_return
       returns[t] = running_return
   ```
6. 学習フィルタリング: `game.max_chain < CURRICULUM[phase].min_chain` のゲームはスキップ
7. 各ステップで目標値を正規化し MSE 損失を計算、Adam で勾配更新:
   ```
   mc_target = (returns[t] - mean) / std_dev  # 正規化
   loss = MSE(V(state_t), mc_target)
   ```
8. `TARGET_UPDATE_INTERVAL` ゲームごとにターゲットネットワークを現在のモデルで更新（一時ファイル経由）
9. 最終モデルを `artifacts/puyo_model_selfplay` に保存

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

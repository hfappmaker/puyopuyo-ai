# AI 評価関数仕様

## 概要

盤面の「良さ」をスコア（`f64`）として数値化する、または最善の配置を直接選択する。AIが配置を比較・選択するための判断基準。

## Evaluator トレイト

```rust
pub trait Evaluator {
    fn find_best_move(&self, board, current, next, next_next) -> Option<(Placement, f64)>;
}
```

唯一のメソッド `find_best_move()` で、各 Evaluator が評価関数と探索戦略の両方を実装する。戻り値は最善配置と評価スコアのタプル。探索深度、評価ロジックは各実装が決定する。

| Evaluator | 探索方式 | 評価関数 |
|-----------|----------|---------|
| `SimulationEvaluator` | depth-1〜2 を BFS 順で統一評価 | max(実連鎖スコア, 仮想ぷよシミュレーション期待値) |
| `NnEvaluator`（Policy-only） | 探索なし、NN 1回推論で直接選択 | Dual Head Network の Policy Head（マスク付き argmax） |
| `NnEvaluator`（MCTS） | MCTS（PUCT探索） | Dual Head Network の Policy + Value Head |

## 共通定数

- `W_GAME_OVER`（`-1000000.0`）: ゲームオーバー状態の盤面に割り当てるスコア。`SimulationEvaluator` で使用

## SimulationEvaluator（仮想ぷよシミュレーション評価）

仮想ぷよを盤面に積んでシミュレーションし、「あと少しで大連鎖になる盤面」を直接評価する評価器。

### 評価の流れ

1. 4色それぞれについて同色2個の `Piece` を作成
2. `enumerate_placements()` で合法配置を列挙（同色Pieceなので最大11配置/色）
3. 各配置に対して `simulate_placement()` → 連鎖解決でスコアを計算
4. 全パターンのスコア合計を配置数で割った期待値（平均スコア）を `f64` として返す
5. 合法配置が1つもない場合（盤面が満杯）は `W_GAME_OVER` を返す

### 仮想ぷよのルール

- 色: Red, Green, Blue, Yellow の4色
- 各色について同色2個の `Piece::new(color, color)` を作成
- `enumerate_placements()` により合法配置のみを評価対象とする（実際のゲーム操作と一致）
- 同色Pieceのため、回転による重複配置は自動的に排除される
- 連鎖解決後にゲームオーバーとなるパターンは、連鎖スコアの代わりに `W_GAME_OVER` をスコアとして加算する

### 計算量

- 最大44回（4色 × 最大11配置）のシミュレーション / 盤面評価
- depth-2 探索と組み合わせた場合: ~484盤面 × 44 ≈ 21,296 シミュレーション / 手

### 探索時の評価値

`find_best_move` の各深度（depth-1〜2）では、`simulate_placement()` が返す実連鎖スコア（`ChainResult.score`）と `simulate_expected_score()` の期待値の大きい方を評価値として採用する:

```
評価値 = max(result.score as f64, simulate_expected_score(&board))
```

これにより、即座に大連鎖が発生する配置を見逃さず、かつ将来の連鎖ポテンシャルも考慮できる。実連鎖スコアは各深度で独立して評価し、深度間で累積しない。

### 用途

`generate-data` バイナリで教師データ生成時に使用。仮想ぷよを落とすことで連鎖の布石をより直接的に評価できる。

## NnEvaluator（Dual Head Network 評価）

Dual Head Network（`PuyoNet`）で盤面とコンテキスト情報（3ツモ）から最善配置を選択する。2つの動作モードを持つ。詳細は `docs/spec/12-nn.md` を参照。

### 動作モード

| モード | 有効化方法 | 探索方式 | 用途 |
|--------|-----------|----------|------|
| Policy-only | デフォルト | 探索なし、1回推論で直接選択 | WASM（ブラウザ） |
| MCTS | `with_mcts(config)` | PUCT探索（Chance Node付き） | self-play、強い推論 |

### 設定メソッド

- `with_mcts(config: MctsConfig)`: MCTSモードを有効化。`MctsConfig` で探索パラメータを指定

### Policy-only モードの評価の流れ

1. 盤面を one-hot エンコーディング（6ch × 14行 × 6列）に変換
2. 3ツモを `context_to_tensor_data()` で24次元ベクトルに変換
3. Dual Head Network の forward pass で (policy_logits, value) を取得
4. `compute_valid_mask()` で合法配置のマスクを生成
5. 不正な配置の logits を `-inf` でマスクし、argmax で最善配置インデックスを選択
6. `index_to_placement()` でインデックスを `Placement`（col, orientation）に変換

### MCTS モードの評価の流れ

1. `mcts_search()` を呼び出し、PUCT探索で配置確率分布 `[f32; 24]` を取得
2. 確率分布から最善配置を選択
3. 詳細は `docs/spec/09-ai-search.md` の MCTS 探索セクションを参照

### 配置インデックス体系

`placement.rs` に定義された、配置と整数インデックス間の双方向マッピング。

| 関数 | 説明 |
|------|------|
| `placement_to_index(placement) -> usize` | `Placement` を 0〜23 のインデックスに変換 |
| `index_to_placement(index) -> Placement` | インデックスを `Placement` に逆変換 |
| `compute_valid_mask(board, piece) -> [bool; 24]` | 合法配置に対応するインデックスを `true` にしたマスク配列を返す |

出力次元は24（6列 × 4方向）で、各インデックスは `col * 4 + orientation` に対応する。

### 特徴

- **2モード対応**: WASM向けの軽量Policy-onlyモードと、self-play向けの高精度MCTSモード
- **mean/std_dev パラメータ不要**: Value Network 時代の z-score 正規化は廃止

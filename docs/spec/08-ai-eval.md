# AI 評価関数仕様

## 概要

盤面の「良さ」をスコア（`f64`）として数値化する。AIが配置を比較・選択するための判断基準。

## Evaluator トレイト

```rust
pub trait Evaluator {
    fn find_best_move(&self, board, current, next, next_next) -> Option<(Placement, f64)>;
}
```

唯一のメソッド `find_best_move()` で、各 Evaluator が評価関数と探索戦略の両方を実装する。戻り値は最善配置と評価スコアのタプル。探索深度、評価ロジックは各実装が決定する。

| Evaluator | 探索深度 | 評価関数 |
|-----------|----------|---------|
| `SimulationEvaluator` | depth-1〜3 を BFS 順で統一評価 | max(実連鎖スコア, 仮想ぷよシミュレーション期待値) |
| `NnEvaluator` | depth-1〜3 を BFS 順で統一評価 | CNN forward pass |

## 共通定数

- `W_GAME_OVER`（`-100000.0`）: ゲームオーバー状態の盤面に割り当てるスコア。`SimulationEvaluator` と `NnEvaluator` の両方で使用

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
- depth-3 探索と組み合わせた場合: ~10,648盤面 × 44 ≈ 468,512 シミュレーション / 手

### 探索時の評価値

`find_best_move` の各深度（depth-1〜3）では、`simulate_placement()` が返す実連鎖スコア（`ChainResult.score`）と `simulate_expected_score()` の期待値の大きい方を評価値として採用する:

```
評価値 = max(result.score as f64, simulate_expected_score(&board))
```

これにより、即座に大連鎖が発生する配置を見逃さず、かつ将来の連鎖ポテンシャルも考慮できる。実連鎖スコアは各深度で独立して評価し、深度間で累積しない。

### 用途

`generate-data` バイナリで教師データ生成時に使用。仮想ぷよを落とすことで連鎖の布石をより直接的に評価できる。

## NnEvaluator（CNN 評価）

CNN（`PuyoValueNet`）で盤面を直接評価する。詳細は `docs/spec/12-nn.md` を参照。

### 評価の流れ

1. ゲームオーバー判定 → ゲームオーバーなら `W_GAME_OVER` を返す
2. 盤面を one-hot エンコーディング（6ch × 14行 × 6列）に変換
3. CNN forward pass で正規化済みスコアを取得
4. z-score 逆変換（`normalized * std_dev + mean`）で生スコアに復元
5. テンソル変換に失敗した場合も `W_GAME_OVER` を返す（安全なフォールバック）

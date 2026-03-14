# AI 評価関数仕様

## 概要

盤面の「良さ」をスコア（`f64`）として数値化する。AIが配置を比較・選択するための判断基準。

## Evaluator トレイト

```rust
pub trait Evaluator {
    fn find_best_move(&self, board, current, next, next_next) -> Option<SearchResult>;
}
```

唯一のメソッド `find_best_move()` で、各 Evaluator が評価関数と探索戦略の両方を実装する。探索深度、連鎖オーバーライドの有無、評価ロジックは各実装が決定する。

| Evaluator | 探索深度 | 連鎖オーバーライド | 評価関数 |
|-----------|----------|-------------------|---------|
| `SimulationEvaluator` | depth-2 → depth-1 | あり | 仮想ぷよシミュレーション |
| `NnEvaluator` | depth-3 → depth-2 → depth-1 | なし | CNN forward pass |

## 共通定数

- `W_GAME_OVER`（`-100000.0`）: ゲームオーバー状態の盤面に割り当てるスコア。`SimulationEvaluator` と `NnEvaluator` の両方で使用

## SimulationEvaluator（仮想ぷよシミュレーション評価）

仮想ぷよを盤面に積んでシミュレーションし、「あと少しで大連鎖になる盤面」を直接評価する評価器。

### 評価の流れ

1. ゲームオーバー判定 → ゲームオーバーなら `W_GAME_OVER` を返す
2. まず現状の盤面で連鎖をチェック（`resolve_chains`）
3. 4色 × 6列 = 24パターンの仮想ぷよ（同色を各列に最大3個縦積み）をシミュレーション
4. 全パターン中の最大連鎖数を `f64` として返す

### 仮想ぷよのルール

- 色: Red, Green, Blue, Yellow の4色
- 各パターンで同色ぷよを最大 `VIRTUAL_PUYO_COUNT`（3）個、同一列に縦積みする
- 列の空きが3未満の場合は入る分だけ積む（空き0ならその列はスキップ）
- row 13 に孤立ぷよがある列は、空きスロット数を `ROWS - 1 - column_height` として計算（孤立ぷよの分を差し引く）

### 計算量

- 24回のシミュレーション / 盤面評価
- depth-2 探索と組み合わせた場合: ~324盤面 × 24 ≈ 7,800 シミュレーション / 手

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

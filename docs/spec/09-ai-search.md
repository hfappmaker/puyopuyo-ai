# AI 探索仕様

## 概要

現在のぷよ組・次のぷよ組・次々のぷよ組の最大3手先まで読み、最も評価の高い配置を「次の一手」として提示する。

## 探索結果

`find_best_move` は `Option<(Placement, f64)>` を返す。最善配置と評価スコアのタプル。配置不能な場合は `None`。

## 配置列挙

各ぷよ組に対して、盤面上の全合法配置を列挙する。

| 方向 | 軸の列範囲 | 条件 | 理由 |
|------|-----------|------|------|
| North | 0〜5 | 到達可能 かつ 列の高さ + 2 ≦ max_rows※ | 軸（下）と衛星（上）が縦に並ぶため2マス必要 |
| South | 0〜5 | 到達可能 かつ 列の高さ + 2 ≦ 13 かつ 回転到達可能※2 | 衛星（下）と軸（上）。軸が14段目に入らないよう制限 |
| East | 0〜4 | 両列到達可能 かつ 軸列の高さ < 13 かつ 衛星列の高さ < sat_max※ | 横並び。軸が14段目に入らないよう制限 |
| West | 1〜5 | 両列到達可能 かつ 軸列の高さ < 13 かつ 衛星列の高さ < sat_max※ | 横並び。軸が14段目に入らないよう制限 |

- 異色の場合: 最大22パターン（North 6 + South 6 + East 5 + West 5）
- 同色の場合: North/South が重複、East(col)/West(col+1) が重複 → 最大11パターン

※ `max_rows` / `sat_max`: 通常は ROWS（14）だが、対象列に row 13 の孤立ぷよ（`has_isolated_top_puyo` が true）がある場合は ROWS - 1（13）に制限される。連鎖消去後に row 13 に残った孤立ぷよを上書きしないための保護。

※2 **回転到達可能性**: South 方向はスポーン時の North から East または West を経由して回転する必要がある。隣接する両列（col-1 と col+1）が共に高さ ≧ 13 の場合、どちらの中間方向への回転もブロックされるため South 配置は除外される。境界列（col=0, col=5）は壁側が常にブロック扱い。

4方向の配置をイテレータチェイン（`filter().map()` + `chain().collect()`）で列挙。同色の重複排除は `normalize_placement` で正規化キーを生成し `HashSet` でフィルタ。

### 配置制限ルール

#### ルール1: 軸ぷよの14段目制限

軸ぷよが14段目（row 13、最上段の非可視行）に着地する配置は不正とする。ゲームプレイでは `FallingPiece` の衝突判定により自然に制限されるが、AI用の配置列挙でも同等の制限を適用する。

- **South**: 軸は衛星の上に位置するため、列の高さが12以上だと軸が14段目に入る → `h + 2 ≦ ROWS - 1`
- **East/West**: 軸列の高さが13以上だと軸が14段目に入る → `軸列の高さ < ROWS - 1`
- **North**: 軸は下側なので `h + 2 ≦ ROWS` で自動的に軸は13段目以下に収まる。ただし row 13 に孤立ぷよがある場合は `max_rows = ROWS - 1` に制限される（衛星の上書き防止）

#### ルール2: スポーン列からの到達可能性

ピースはスポーン列（列2）の上部から出現し、左右移動で他の列に到達する。高さが ROWS - 1（13）以上の列は上部が塞がれているため、通過できない。

`compute_reachable_columns` 関数が `SPAWN_COL` から左右に展開し、到達可能な列を計算する。イテレータチェイン（`once().chain().chain()` + `take_while`）で実装:
- `SPAWN_COL` から左方向: 高さ ≧ 13 の列で遮断（`take_while`）
- `SPAWN_COL` から右方向: 同上
- 高さ ≧ 13 の列自体も到達不可

East/West 配置では軸列・衛星列の両方が到達可能でなければならない。

#### ルール3: row 13 孤立ぷよとの重複配置防止

連鎖消去により row 13 にぷよが孤立して残る場合がある（下の行が消えても `apply_gravity` は row 13 を移動しない）。この孤立ぷよを上書きしないよう、3層の防御を行う:

1. **AI配置列挙**: `column_info(col)` の孤立フラグで検出し、North/East/West の衛星配置先の上限を `ROWS - 1` に制限
2. **ゲームプレイ衝突判定**: `FallingPiece::can_occupy` でセルレベルの衝突チェックを実施（`board.get(col, row).is_color()` で占有セルを検出）
3. **place_piece 防御**: North 配置時、衛星の着地先セルが占有済みなら `drop_puyo` をスキップ

## find_best_move

`find_best_move` は `Evaluator` トレイトの唯一のメソッド。各評価器が評価関数・探索深度を含む探索戦略を完全に実装する。

```rust
fn find_best_move(&self, board: &Board, current: &Piece, next: &Piece, next_next: &Piece) -> Option<(Placement, f64)>
```

| Evaluator | 探索深度 |
|-----------|----------|
| `SimulationEvaluator` | depth-1〜2 を BFS 順で統一評価 |
| `NnEvaluator`（Policy-only） | 探索なし、NN 1回推論で直接選択 |
| `NnEvaluator`（MCTS） | MCTS（PUCT探索） |

## 探索の流れ

1. 現在のぷよ組の全配置パターンを列挙する
2. 各配置に対して盤面をシミュレーション（設置 + 連鎖解決）する
3. ゲームオーバーになる配置はスキップする
4. BFS 順（depth-1 → depth-2）で全深度の盤面を評価し、単一の `best_score` / `best_placement` を更新する
5. 全深度を通じて最高評価を得た1手目の配置を返す

フォールバック分岐は不要。depth-2 で有効な盤面がなくても、depth-1 の評価結果がすでに `best_score` に反映されているため、自然に浅い深度の最良手が選ばれる。

### 計算量

- depth-2: 最大 22 × 22 = 484 盤面の評価

## MCTS 探索（mcts.rs）

PUCT（Predictor Upper Confidence bounds applied to Trees）に基づくモンテカルロ木探索。`NnEvaluator` の MCTSモードで使用される。

### 構造体

| 構造体 | 説明 |
|--------|------|
| `MctsTree` | 探索木全体を管理。ルートノードから探索を実行 |
| `MctsNode` | 探索木の各ノード。訪問回数・累積価値・prior・子ノード等を保持。`priors: [f32; 24]` にNN展開時のpolicy出力を保存し、PUCT選択・子ノード作成時に参照する |

### ランダムツモの扱い

3手先以降のツモが不明な場合、決定論的なハッシュ関数（`sample_piece`）でランダムツモを生成する。ノードIDとアクションIDをシードとして使用するため、同じ探索状態では常に同じツモが生成される。

### Min-Max Value Normalization（MuZero Reanalyze方式）

PUCT計算時にQ値を [0, 1] に正規化する。探索木内で観測されたValue（NNの推論値）の最小値・最大値を `MctsTree` で追跡し、以下の式で正規化:

```
Q_normalized = (Q - Q_min) / (Q_max - Q_min)
```

- `Q_min == Q_max`（まだ情報がない場合）は 0.5 を返す
- min/max は Backpropagation 時に毎回更新される

これにより、Q項（活用）と Prior項（探索）のスケールが揃い、`c_puct` のチューニングがスコアレンジに依存しなくなる。

### 探索の流れ

1. ルートノードから PUCT で最も有望な子ノードを選択（Selection）。各ノードの `priors` フィールドに保存されたNN policy出力を事前確率として使用。Q値は Min-Max 正規化して [0, 1] に変換
2. 未展開ノードに到達したら、`PuyoNet` の forward pass で (policy_logits, value) を取得（Expansion + Evaluation）
3. Policy logits を masked softmax でアクション確率に変換し、ノードの `priors` フィールドに保存。既存の子ノードの `prior` も更新
4. Value を探索パスに沿って逆伝播（Backpropagation）。同時に min/max を更新
5. 規定回数の反復後、ルート直下の訪問回数分布を返す

### API

```rust
pub fn mcts_search(
    board: &Board,
    current: &Piece,
    next: &Piece,
    next_next: &Piece,
    model: &PuyoNet<NdArray>,
    device: &<NdArray as Backend>::Device,
    num_simulations: usize,
    c_puct: f32,
    temperature: f32,
) -> [f32; 24]
```

- **入力**: 盤面、3ツモ（current, next, next_next）、NNモデル、デバイス、探索パラメータ（シミュレーション回数、PUCT定数、温度）
- **出力**: 24次元の確率分布（各配置の訪問回数に基づく）

## 共通ユーティリティ（placement.rs）

- `simulate_placement(board, piece, placement)`: 配置シミュレーション。一時的な `GameState` でピースを設置し連鎖解決。結果の盤面と `ChainResult` を返す。元の盤面は変更されない
- `enumerate_placements(board, piece)`: 盤面上の全合法配置を列挙する

2手目以降の探索は各 Evaluator が `find_best_move` 内にインラインで実装する（共通の再帰関数は使用しない）。

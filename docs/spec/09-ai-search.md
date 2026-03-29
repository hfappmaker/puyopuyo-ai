# AI 探索仕様

## 概要

現在のぷよ組・次のぷよ組・次々のぷよ組の最大3手先まで読み、最も評価の高い配置を「次の一手」として提示する。

## 探索結果

`find_best_move` は `Option<(Placement, f64)>` を返す。最善配置と評価スコアのタプル。配置不能な場合は `None`。

## 配置列挙

各ぷよ組に対して、盤面上の全合法配置を列挙する。

| 方向 | 軸の列範囲 | 条件 | 理由 |
|------|-----------|------|------|
| North | 0〜COLS-1 | 到達可能 かつ 列の高さ + 2 ≦ max_rows※ | 軸（下）と衛星（上）が縦に並ぶため2マス必要 |
| South | 0〜COLS-1 | 到達可能 かつ 列の高さ + 2 ≦ ROWS-1 かつ 回転到達可能※2 | 衛星（下）と軸（上）。軸が最上非可視行に入らないよう制限 |
| East | 0〜COLS-2 | 両列到達可能 かつ 軸列の高さ < ROWS-1 かつ 衛星列の高さ < sat_max※ | 横並び。軸が最上非可視行に入らないよう制限 |
| West | 1〜COLS-1 | 両列到達可能 かつ 軸列の高さ < ROWS-1 かつ 衛星列の高さ < sat_max※ | 横並び。軸が最上非可視行に入らないよう制限 |

- 異色の場合: 最大10パターン（COLS=3: North 3 + South 3 + East 2 + West 2）
- 同色の場合: North/South が重複、East(col)/West(col+1) が重複 → 最大5パターン

※ `max_rows` / `sat_max`: 通常は ROWS（8）だが、対象列に最上非可視行（row ROWS-1）の孤立ぷよ（`has_isolated_top_puyo` が true）がある場合は ROWS - 1（7）に制限される。連鎖消去後に最上非可視行に残った孤立ぷよを上書きしないための保護。

※2 **回転到達可能性**: South 方向はスポーン時の North から East または West を経由して回転する必要がある。隣接する両列（col-1 と col+1）が共に高さ ≧ ROWS-1 の場合、どちらの中間方向への回転もブロックされるため South 配置は除外される。境界列（col=0, col=COLS-1）は壁側が常にブロック扱い。

4方向の配置をイテレータチェイン（`filter().map()` + `chain().collect()`）で列挙。同色の重複排除は `normalize_placement` で正規化キーを生成し `HashSet` でフィルタ。

### 配置制限ルール

#### ルール1: 軸ぷよの最上非可視行制限

軸ぷよが最上非可視行（row ROWS-1）に着地する配置は不正とする。ゲームプレイでは `FallingPiece` の衝突判定により自然に制限されるが、AI用の配置列挙でも同等の制限を適用する。

- **South**: 軸は衛星の上に位置するため、列の高さが VISIBLE_ROWS 以上だと軸が最上非可視行に入る → `h + 2 ≦ ROWS - 1`
- **East/West**: 軸列の高さが ROWS-1 以上だと軸が最上非可視行に入る → `軸列の高さ < ROWS - 1`
- **North**: 軸は下側なので `h + 2 ≦ ROWS` で自動的に軸は ROWS-2 段目以下に収まる。ただし最上非可視行に孤立ぷよがある場合は `max_rows = ROWS - 1` に制限される（衛星の上書き防止）

#### ルール2: スポーン列からの到達可能性

ピースはスポーン列（列1）の上部から出現し、左右移動で他の列に到達する。高さが ROWS - 1（7）以上の列は上部が塞がれているため、通過できない。

`compute_reachable_columns` 関数が `SPAWN_COL` から左右に展開し、到達可能な列を計算する。イテレータチェイン（`once().chain().chain()` + `take_while`）で実装:
- `SPAWN_COL` から左方向: 高さ ≧ ROWS-1 の列で遮断（`take_while`）
- `SPAWN_COL` から右方向: 同上
- 高さ ≧ ROWS-1 の列自体も到達不可

East/West 配置では軸列・衛星列の両方が到達可能でなければならない。

#### ルール3: row 13 孤立ぷよとの重複配置防止

連鎖消去により最上非可視行（row ROWS-1）にぷよが孤立して残る場合がある（下の行が消えても `apply_gravity` は最上非可視行を移動しない）。この孤立ぷよを上書きしないよう、3層の防御を行う:

1. **AI配置列挙**: `column_info(col)` の孤立フラグで検出し、North/East/West の衛星配置先の上限を `ROWS - 1` に制限
2. **ゲームプレイ衝突判定**: `FallingPiece::can_occupy` でセルレベルの衝突チェックを実施（`board.get(col, row).is_color()` で占有セルを検出）
3. **place_piece 防御**: North 配置時、衛星の着地先セルが占有済みなら `drop_puyo` をスキップ

## find_best_move

`find_best_move` は `Evaluator<G: Game>` トレイトの主要メソッド。各評価器が評価関数・探索深度を含む探索戦略を完全に実装する。

```rust
fn find_best_move(&self, state: &G::State) -> Option<(Placement, f64)>
```

ぷよぷよの場合、`G::State` は `PuyoState`（board + 3 pieces）。

| Evaluator | 探索深度 |
|-----------|----------|
| `SimulationEvaluator` | depth-1〜2 を BFS 順で統一評価 |
| `NnEvaluator`（Policy-only） | 探索なし、NN 1回推論で直接選択 |
| `NnEvaluator`（MCTS） | Gumbel MCTS（Sequential Halving + PUCT） |

## 探索の流れ

1. 現在のぷよ組の全配置パターンを列挙する
2. 各配置に対して盤面をシミュレーション（設置 + 連鎖解決）する
3. ゲームオーバーになる配置はスキップする
4. BFS 順（depth-1 → depth-2）で全深度の盤面を評価し、単一の `best_score` / `best_placement` を更新する
5. 全深度を通じて最高評価を得た1手目の配置を返す

フォールバック分岐は不要。depth-2 で有効な盤面がなくても、depth-1 の評価結果がすでに `best_score` に反映されているため、自然に浅い深度の最良手が選ばれる。

### 計算量

- depth-2: 最大 10 × 10 = 100 盤面の評価（COLS=3 の場合）

## Gumbel MCTS 探索（mcts.rs）

Gumbel AlphaZero（Danihelka et al. 2022）に基づくモンテカルロ木探索。ルートでSequential Halving + Gumbel-Top-kを使用し、少ないシミュレーション数（32〜64）でも高品質な手選択とimproved policyターゲットを生成する。内部ノードではPUCT選択を使用する。

### 構造体

| 構造体 | 説明 |
|--------|------|
| `MctsTree<G: Game>` | 探索木全体を管理。`Game` トレイトでジェネリック化。`gamma: f32` で割引率、`root_value: f32` でルートの価値推定、`min_value`/`max_value` でMin-Max正規化範囲を保持 |
| `MctsNode` | 探索木の各ノード。`visit_count: u32`、`total_value: f32`、`prior: f32`、`priors: Vec<f32>`、`logits: Vec<f32>`、`children: Vec<Option<usize>>`、`expanded: bool`、`terminal: bool`、`immediate_reward: f32`（連鎖スコア）、`valid_mask: Vec<bool>`、`depth: u32` を保持。固定長配列から `Vec` に変更されゲーム非依存化 |

`MctsTree<G: Game>` の公開メソッド:

| メソッド | 説明 |
|---------|------|
| `new(state: &G::State, gamma) -> Self` | 探索木を初期化。`G::State`（ぷよぷよの場合は `PuyoState`）からルートノードを作成 |
| `root_q_values() -> Vec<f32>` | ルート直下の各アクションの平均累積報酬（Q値）を返す |

### ランダムツモの扱い

3手先以降のツモが不明な場合、決定論的なハッシュ関数（`sample_piece`）でランダムツモを生成する。ノードID、アクションID、および配置後の盤面のFNV-1aハッシュ（`board_hash`）をシードとして使用するため、同じ盤面状態・同じアクションでは常に同じツモが生成される。`board_hash` と `sample_piece` は `puyo-ai/src/puyo_game.rs` に定義されている（`PuyoGame` の `Game` トレイト実装の一部）。

### Min-Max Value Normalization（MuZero Reanalyze方式）

内部ノードのPUCT計算時にQ値を [0, 1] に正規化する。探索木内で観測されたValueの最小値・最大値を `MctsTree` で追跡し、以下の式で正規化:

```
Q_normalized = (Q - Q_min) / (Q_max - Q_min)
```

- `Q_min == Q_max`（まだ情報がない場合）は 0.5 を返す
- 正規化結果は `[0.0, 1.0]` にクランプされる

### 探索の流れ（Gumbel Sequential Halving）

1. **ルート展開**: NN forward pass で logits と value を取得。logits はノードに保存
2. **Gumbelノイズサンプリング**: 各有効アクションに Gumbel(0,1) ノイズ g(a) を付与
3. **初期スコア計算**: `score(a) = g(a) + logit(a)` でTop-m アクションを選択
4. **Sequential Halving**: 各フェーズで残りアクションにシミュレーションを均等割当 → `simulate_from_root_action` でルートアクションを強制して探索 → completed Q-values と sigma_bar でスコア更新 → 上位半分を残す
5. **残り予算消化**: 生存アクションに残りシミュレーションを投入
6. **Improved Policy計算**: `π_improved(a) ∝ π(a) · exp(advantage(a) · c_visit)` で改善されたポリシーターゲットを生成

内部ノード（ルート以外）では動的PUCT選択を使用する。探索定数 `c_puct` は親ノードの訪問回数 `N(s)` に応じて対数的に増加する:

```
c(s) = log((1 + N(s) + c_puct_base) / c_puct_base) + c_puct_init
```

これにより、探索序盤は `c_puct_init` に近い値で活用寄りに、訪問回数が増えるにつれて緩やかに探索寄りになる。`c_puct_base` が大きいほど変動が小さく安定する。

### Completed Q-values

Gumbel探索の核心概念。ルートの各アクションについて:

- **訪問済みアクション**: 実際のツリーQ値 `total_value / visit_count` を使用
- **未訪問アクション**: ルートの価値推定 `root_value` をプロキシとして使用

これにより、少ないシミュレーションでも全アクションの比較が可能になる。

### sigma_bar（スコア更新）

Sequential Halvingの各フェーズでスコアを更新する際に使用:

```
sigma_bar(a) = (c_visit + N_max) × q_normalized(a)
```

ここで `N_max` はルート子ノードの最大訪問回数、`q_normalized` はcompleted Q-valuesのmin-max正規化値。

### Gumbelノイズ

標準Gumbel(0,1)分布からサンプリング: `g = -log(-log(u))` (u ~ Uniform(0,1))。既存の `xorshift64_f64` PRNGを使用。Gumbelノイズにより探索の多様性が確保されるため、Dirichletノイズは不要。

### InferenceProvider トレイト

MCTS がNN推論バックエンドに依存しないよう抽象化するトレイト。詳細は `docs/spec/08-ai-eval.md` を参照。

```rust
pub trait InferenceProvider {
    fn infer(&self, board_data: &[f32], context_data: &[f32]) -> (Vec<f32>, f32);
}
```

### API

```rust
pub fn mcts_search<G: Game>(
    state: &G::State,
    provider: &dyn InferenceProvider,
    config: &MctsConfig,
    seed: u64,
) -> (Vec<f32>, Vec<f32>)
```

- **入力**: `G::State`（ぷよぷよの場合は `PuyoState`）、`InferenceProvider`（推論プロバイダ）、`MctsConfig`（探索パラメータ一式）、Gumbelシード
- **出力**: (NUM_ACTIONS次元のimproved policy, NUM_ACTIONS次元のQ値)。固定長配列から `Vec<f32>` に変更

`MctsConfig`のフィールド:
- `num_simulations`: シミュレーション回数（デフォルト64）
- `c_puct_init`: 動的PUCT初期値（デフォルト1.5）
- `c_puct_base`: 動的PUCTベース定数（デフォルト19652.0）
- `m`: 初期にGumbel-Top-kで選択するアクション数（デフォルト16）
- `c_visit`: advantageのスケーリング係数（デフォルト5.0）
- `gamma`: 将来報酬の割引率（デフォルト0.95）

## 共通ユーティリティ（placement.rs）

- `simulate_placement(board, piece, placement) -> (Board, ChainResult)`: 配置シミュレーション。一時的な `GameState` でピースを設置し連鎖解決。結果の盤面と `ChainResult` を返す。元の盤面は変更されない
- `enumerate_placements(board, piece) -> Vec<Placement>`: 盤面上の全合法配置を列挙する
- `NUM_ACTIONS: usize = COLS * 4`: 配置インデックスの総数（3列 × 4方向 = 12（COLS=3 の場合））。`puyo_core::config` に定義され、`placement.rs` から再エクスポート
- `placement_to_index(placement) -> usize`: `Placement` を 0〜NUM_ACTIONS-1 のインデックスに変換（`col * 4 + orientation.as_u8()`）
- `index_to_placement(index) -> Placement`: インデックスを `Placement` に逆変換。`index >= NUM_ACTIONS` でパニック
- `compute_valid_mask(board, piece) -> [bool; NUM_ACTIONS]`: 合法配置に対応するインデックスを `true` にしたマスク配列を返す

## ハッシュユーティリティ

### hash_util.rs

- `splitmix64(s: u64) -> u64`: splitmix64 finalizer。シード値を分散の良いハッシュに変換する。MCTS 内の `sample_piece` で使用

### puyo_game.rs 内のハッシュ関数

- `board_hash(board: &Board) -> u64`: 盤面の FNV-1a ハッシュ。MCTS のランダムツモ生成シードおよび `NnEvaluator` の Gumbel シードとして使用。`puyo-ai/src/puyo_game.rs` に定義（`mcts.rs` から移動）

2手目以降の探索は各 Evaluator が `find_best_move` 内にインラインで実装する（共通の再帰関数は使用しない）。

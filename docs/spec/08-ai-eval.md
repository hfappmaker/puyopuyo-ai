# AI 評価関数仕様

## 概要

盤面の「良さ」をスコア（`f64`）として数値化する、または最善の配置を直接選択する。AIが配置を比較・選択するための判断基準。

## Evaluator トレイト

```rust
pub trait Evaluator<G: Game> {
    fn find_best_move(&self, state: &G::State) -> Option<(G::Action, f64)>;
    fn set_num_simulations(&mut self, _num_simulations: usize) {}  // デフォルト実装: 何もしない
}
```

`Game` トレイト（`az-framework` クレート）でターン制ゲームを抽象化し、`Evaluator` はジェネリックパラメータ `G: Game` を取る。主要メソッド `find_best_move()` は `&G::State`（ぷよぷよの場合は `&PuyoState`）を受け取り、最善アクションと評価スコアのタプルを返す（ぷよ��よの場合 `G::Action` = `Placement`）。探索深度、評価ロジックは各実装が決定する。`set_num_simulations()` は MCTS のシミュレーション数を動的に変更するためのメソッド（デフォルト実装は何もしない）。

| Evaluator | Game実装 | 探索方式 | 評価関数 |
|-----------|---------|----------|---------|
| `SimulationEvaluator` | `Evaluator<PuyoGame>` | depth-1〜2 を BFS 順で統一評価 | max(実連鎖スコア, 仮想ぷよシミュレーション期待値) |
| `NnEvaluator`（Policy-only） | `Evaluator<PuyoGame>` | 探索なし、NN 1回推論で直接選択 | Dual Head Network の Policy Head（マスク付き argmax） |
| `NnEvaluator`（MCTS） | `Evaluator<PuyoGame>` | Gumbel MCTS（Sequential Halving + PUCT） | Dual Head Network の Policy + Value Head |

## モジュール構成

### az-framework（汎用ゲームAI）

```
az-framework/src/
├── lib.rs                # 公開モジュール宣言
├── eval.rs               # Evaluator<G: Game> トレイト
├── model.rs              # [nn] GameModel<B: Backend> トレイト
├── mcts.rs               # [nn] Gumbel MCTS 探索（MctsTree<G: Game>, InferenceProvider トレイト）
├── nn_eval.rs            # [nn] MctsConfig + DirectInference<B, M: GameModel<B>>
└── inference_server.rs   # [nn] GPU バッチ推論サーバー（InferenceClient）
```

### puyo-player（ぷよぷよ固有AI）

```
puyo-player/src/
├── lib.rs                # 公開モジュール宣言 + az-framework/puyo-core 再エクスポート
├── eval.rs               # SimulationEvaluator（Evaluator<PuyoGame> 実装）
├── puyo_game.rs          # PuyoGame（Game トレイト実装）
└── nn_eval.rs            # [nn] PuyoGameModel<B> + NnEvaluator（Evaluator<PuyoGame> 実装）
```

`[nn]` マーク付きモジュールは `#[cfg(feature = "nn")]` で条件コンパイルされる。

## 共通定数

- `W_GAME_OVER`（`-1000000.0`）: ゲームオーバー状態の盤面に割り当てるスコア。`SimulationEvaluator` で使用

## SimulationEvaluator（仮想ぷよシミュレーション評価）

仮想ぷよを盤面に積んでシミュレーションし、「あと少しで大連鎖になる盤面」を直接評価する評価器。

### 評価の流れ

1. `PuyoColor::all_colors()` で取得したアクティブな各色について同色2個の `Piece` を作成（NUM_COLORS=3 なら3色）
2. `enumerate_placements()` で合法配置を列挙（同色Pieceなので最大5配置/色（3列の場合））
3. 各配置に対して `simulate_placement()` → 連鎖解決でスコアを計算
4. 全パターンのスコア合計を配置数で割った期待値（平均スコア）を `f64` として返す
5. 合法配置が1つもない場合（盤面が満杯）は `W_GAME_OVER` を返す

### 仮想ぷよのルール

- 色: `PuyoColor::all_colors()` で取得（NUM_COLORS=3 なら Red, Green, Blue）
- 各色について同色2個の `Piece::new(color, color)` を作成
- `enumerate_placements()` により合法配置のみを評価対象とする（実際のゲーム操作と一致）
- 同色Pieceのため、回転による重複配置は自動的に排除される
- 連鎖解決後にゲームオーバーとなるパターンは、連鎖スコアの代わりに `W_GAME_OVER` をスコアとして加算する

### 計算量

- 最大15回（3色 × 最大5配置（3列の場合））のシミュレーション / 盤面評価
- depth-2 探索と組み合わせた場合: ~100盤面 × 15 ≈ 1,500 シミュレーション / 手

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
| MCTS | `with_mcts(config)` | Gumbel MCTS（Sequential Halving + PUCT） | self-play、強い推論 |

### 設定メソッド

- `with_mcts(config: MctsConfig)`: MCTSモードを有効化。`MctsConfig` で探索パラメータを指定（`num_simulations`, `c_puct_init`, `c_puct_base`, `m`, `c_visit`, `gamma`）
- `set_num_simulations(num_simulations: usize)`: MCTS シミュレーション数を動的に変更する

### Policy-only モードの評価の流れ

1. 盤面を one-hot エンコーディング（5ch × 8行 × 3列）に変換
2. 3ツモを `context_to_tensor_data()` で18次元ベクトルに変換
3. Dual Head Network の forward pass で (policy_logits, value) を取得
4. `compute_valid_mask()` で合法配置のマスクを生成
5. 不正な配置の logits を `-inf` でマスクし、argmax で最善配置インデックスを選択
6. `index_to_placement()` でインデックスを `Placement`（col, orientation）に変換

### MCTS モードの評価の流れ

1. `mcts_search()` を呼び出し、Gumbel MCTS（Sequential Halving + PUCT）で improved policy `[f32; NUM_ACTIONS]` を取得。`gamma` 引数で将来報酬の割引率を指定する
2. improved policy から最善配置を選択
3. 詳細は `docs/spec/09-ai-search.md` の Gumbel MCTS 探索セクションを参照

### 配置インデックス体系

`placement.rs` に定義された、配置と整数インデックス間の双方向マッピング。

| 関数 | 説明 |
|------|------|
| `placement_to_index(placement) -> usize` | `Placement` を 0〜NUM_ACTIONS-1 のインデックスに変換 |
| `index_to_placement(index) -> Placement` | インデックスを `Placement` に逆変換 |
| `compute_valid_mask(board, piece) -> [bool; NUM_ACTIONS]` | 合法配置に対応するインデックスを `true` にしたマスク配列を返す |

出力次元は NUM_ACTIONS（COLS × 4 = 12（3列の場合））で、各インデックスは `col * 4 + orientation` に対応する。

### InferenceProvider トレイト

MCTS が NN バックエンドに依存しないよう、推論を抽象化するトレイト（`mcts.rs` に定義）。

```rust
pub trait InferenceProvider {
    /// Returns (logits[NUM_ACTIONS], value_transformed).
    /// value は inverse-transform 済み（生の累積報酬スケール）。
    fn infer(&self, board_data: &[f32], context_data: &[f32]) -> (Vec<f32>, f32);
}
```

| 実装 | 説明 |
|------|------|
| `DirectInference<B: Backend, M: GameModel<B>>` | 単一サンプル推論。NdArray バックエンド（CPU）で使用。`NnEvaluator` 内部で `PuyoGameModel` と共に保持 |
| `InferenceClient` | GPU バッチ推論サーバーへのクライアント。`inference_server.rs` で定義。`Clone` 可能で各ゲームスレッドに配布 |

### GPU バッチ推論サーバー（inference_server.rs）

`start_inference_server<B: Backend, M: GameModel<B>>(model: M, device, max_batch_size) -> InferenceClient`

- GPU スレッドがモデルを保持し、バッチ forward pass を実行
- N 個のゲームスレッドが `InferenceClient` 経由でリクエストを送信・ブロック
- greedy バッチング: 最初のリクエスト到着後、`max_batch_size` まで `try_recv` で追加収集

### 特徴

- **2モード対応**: WASM向けの軽量Policy-onlyモードと、self-play向けの高精度MCTSモード
- **mean/std_dev パラメータ不要**: Value Network 時代の z-score 正規化は廃止

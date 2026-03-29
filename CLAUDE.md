# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

日本語でコミュニケーションすること。

## ビルド・テスト

```bash
cargo build                          # Rustビルド（全クレート）
cargo test --workspace               # 全テスト実行
cargo test -p puyo-core              # 単一クレートのテスト
cargo test -p puyo-player test_name   # 単一テスト実行
bash scripts/build-wasm.sh           # WASMビルド（テスト→WASM→npm install）
cd web && npm run dev                # フロントエンド開発サーバー
cd web && npm run build              # フロントエンドプロダクションビルド（tsc + vite build）
```

WASMの手動ビルド: `wasm-pack build crates/puyo/puyo-wasm --target web --out-dir ../../../web/wasm-pkg`

Rust側を変更したらWASM再ビルドが必要。Viteキャッシュが残る場合は `rm -rf web/node_modules/.vite`。

## プロジェクト構造

Rustワークスペース（`crates/`配下）+ TypeScript フロントエンド（`web/`）。

```
crates/
├── az-framework/          # 汎用 AlphaZero フレームワーク（ゲーム非依存）
└── puyo/                  # ぷよぷよ関連
    ├── puyo-core/
    ├── puyo-nn/
    ├── puyo-player/
    ├── puyo-trainer/
    └── puyo-wasm/
```

| クレート | 役割 |
|---------|------|
| `az-framework` | AlphaZeroフレームワーク（Game trait, Evaluator trait, GameModel trait, MCTS, DirectInference, 推論サーバー, AlphaZeroDataset, value_transform）。ゲーム非依存 |
| `puyo-core` | ゲームエンジン（Board（連鎖解決含む）, GameState, Piece, Score, PuyoState, random_piece, エンコーディング関数, placement（配置列挙・シミュレーション）） |
| `puyo-player` | ぷよぷよ固有AI（SimulationEvaluator, NnEvaluator, PuyoGameModel, PuyoGame） |
| `puyo-nn` | CNN Dual Head ネットワーク（PuyoNet: Policy + Value） |
| `puyo-trainer` | 学習パイプライン（3つのバイナリ: generate-data, train, self-play） |
| `puyo-wasm` | WASMブリッジ（wasm-bindgen, WasmGame struct） |

依存方向: `az-framework` ← `puyo-player` ← `puyo-wasm`、`puyo-core` ← `puyo-nn` ← `puyo-player`（`nn` feature有効時）、`puyo-core`/`puyo-player`/`puyo-nn` ← `puyo-trainer`

## アーキテクチャの要点

### Evaluator trait（多態性の中心）
`az-framework/src/eval.rs`の`Evaluator<G: Game>`トレイト（`find_best_move(&G::State) -> Option<(Placement, f64)>`）がAIの核。`Game` トレイト（`az-framework/src/game.rs`）でターン制ゲームを抽象化し、`PuyoGame`（`puyo-player/src/puyo_game.rs`）がぷよぷよ用の `Game` 実装を提供する。`PuyoState`（`puyo-core/src/state.rs`）はAI用の軽量ゲーム状態（board + 3 pieces）とエンコーディング関数を含む。
- `SimulationEvaluator`（`puyo-player/src/eval.rs`）: `Evaluator<PuyoGame>` を実装。仮想ぷよシミュレーションで盤面を評価（2手先読みBFS）
- `NnEvaluator`（`puyo-player/src/nn_eval.rs`、`nn` feature flag有効時のみ）: `Evaluator<PuyoGame>` を実装。Dual Head Network（Policy + Value）で評価。MCTSモード（`MctsConfig`付き、PUCT探索）とPolicy-onlyモード（WASM用、1回推論）の2モード

### GameModel trait（NN抽象化）
`az-framework/src/model.rs`の`GameModel<B: Backend>`トレイトがNNモデルを抽象化。`board_shape()`、`context_size()`、`num_actions()`、`forward()`、`postprocess_value()`を定義し、`DirectInference`と`inference_server`がゲーム非依存で動作する。`PuyoGameModel`（`puyo-player/src/nn_eval.rs`）が`PuyoNet`をラップして実装。

### Feature flags
- `az-framework`の`nn`フィーチャーフラグでNN関連モジュール（`model`, `mcts`, `nn_eval`, `inference_server`）と`burn`依存を制御。
- `puyo-player`の`nn`フィーチャーフラグで`az-framework/nn` + `puyo-nn` + `burn`を有効化。`puyo-wasm`は`nn`を有効にしてビルド。
- `puyo-player`の`cuda`フィーチャーフラグ（`nn` + `burn/cuda-jit`）でGPU推論を有効化。
- `puyo-trainer`の`gpu`（デフォルト）/`cpu`フィーチャーフラグでバックエンドを切り替え。`gpu`は`burn/cuda-jit`+`burn/fusion`+`burn/autotune`を有効にする。

詳細は仕様書を参照: NN → `docs/spec/12-nn.md`、WASM → `docs/spec/10-wasm-bridge.md`、学習 → `docs/spec/13-trainer.md`、フロントエンド → `docs/spec/11-frontend.md`

## モデルファイル

- `artifacts/puyo_model.bin` (~500KB): 学習済みモデル（`BinFileRecorder`形式）
- `web/public/models/`: ブラウザ用デプロイ先

## ドキュメント更新ルール

**ソースファイルを変更したら、必ず対応する `docs/spec/` のドキュメントも更新すること。**

| 変更したファイル | 更新すべきドキュメント |
|----------------|----------------------|
| `crates/puyo/puyo-core/src/board.rs` | `docs/spec/02-board.md`, `docs/spec/01-architecture.md` |
| `crates/puyo/puyo-core/src/piece.rs` | `docs/spec/03-piece.md`, `docs/spec/01-architecture.md` |
| `crates/puyo/puyo-core/src/score.rs` | `docs/spec/05-score.md`, `docs/spec/01-architecture.md` |
| `crates/puyo/puyo-core/src/game.rs` | `docs/spec/06-game.md`, `docs/spec/01-architecture.md` |
| `crates/puyo/puyo-core/src/rand.rs` | `docs/spec/07-rng.md`, `docs/spec/01-architecture.md` |
| `crates/puyo/puyo-core/src/state.rs` | `docs/spec/08-ai-eval.md`, `docs/spec/12-nn.md` |
| `crates/puyo/puyo-core/src/placement.rs` | `docs/spec/09-ai-search.md` |
| `crates/az-framework/src/eval.rs`, `crates/az-framework/src/model.rs`, `crates/az-framework/src/nn_eval.rs`, `crates/az-framework/src/inference_server.rs` | `docs/spec/08-ai-eval.md` |
| `crates/puyo/puyo-player/src/eval.rs`, `crates/puyo/puyo-player/src/nn_eval.rs`, `crates/puyo/puyo-player/src/puyo_game.rs` | `docs/spec/08-ai-eval.md` |
| `crates/az-framework/src/mcts.rs` | `docs/spec/09-ai-search.md` |
| `crates/puyo/puyo-wasm/src/lib.rs` | `docs/spec/10-wasm-bridge.md` |
| `web/src/*.ts` | `docs/spec/11-frontend.md` |
| `crates/puyo/puyo-nn/src/*.rs` | `docs/spec/12-nn.md` |
| `crates/puyo/puyo-trainer/src/**/*.rs` | `docs/spec/13-trainer.md` |

## 仕様書

詳細な仕様は `docs/spec/` 配下（00〜13の各ドキュメント）を参照。

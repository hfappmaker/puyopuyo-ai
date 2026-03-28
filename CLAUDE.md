# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

日本語でコミュニケーションすること。

## ビルド・テスト

```bash
cargo build                          # Rustビルド（全クレート）
cargo test --workspace               # 全テスト実行
cargo test -p puyo-core              # 単一クレートのテスト
cargo test -p puyo-ai test_name      # 単一テスト実行
bash scripts/build-wasm.sh           # WASMビルド（テスト→WASM→npm install）
cd web && npm run dev                # フロントエンド開発サーバー
cd web && npm run build              # フロントエンドプロダクションビルド（tsc + vite build）
```

WASMの手動ビルド: `wasm-pack build crates/puyo-wasm --target web --out-dir ../../web/wasm-pkg`

Rust側を変更したらWASM再ビルドが必要。Viteキャッシュが残る場合は `rm -rf web/node_modules/.vite`。

## プロジェクト構造

Rustワークスペース（`crates/`配下）+ TypeScript フロントエンド（`web/`）。

| クレート | 役割 |
|---------|------|
| `puyo-core` | ゲームエンジン（Board（連鎖解決含む）, GameState, Piece, Score, RNG） |
| `puyo-ai` | AI探索・評価（Evaluator trait, SimulationEvaluator, NnEvaluator, MCTS, find_best_move） |
| `puyo-nn` | CNN Dual Head ネットワーク（PuyoNet: Policy + Value, one-hot encoding） |
| `puyo-trainer` | 学習パイプライン（3つのバイナリ: generate-data, train, self-play） |
| `puyo-wasm` | WASMブリッジ（wasm-bindgen, WasmGame struct） |

依存方向: `puyo-core` ← `puyo-ai` ← `puyo-wasm`、`puyo-core` ← `puyo-nn` ← `puyo-ai`（`nn` feature有効時）、`puyo-core`/`puyo-ai`/`puyo-nn` ← `puyo-trainer`

## アーキテクチャの要点

### Evaluator trait（多態性の中心）
`puyo-ai/src/eval.rs`の`Evaluator`トレイト（`find_best_move(&Board, &Piece, &Piece, &Piece) -> Option<(Placement, f64)>`）がAIの核。
- `SimulationEvaluator`: 仮想ぷよシミュレーションで盤面を評価（2手先読みBFS）
- `NnEvaluator`（`puyo-ai/src/nn_eval.rs`、`nn` feature flag有効時のみ）: Dual Head Network（Policy + Value）で評価。MCTSモード（`MctsConfig`付き、PUCT探索）とPolicy-onlyモード（WASM用、1回推論）の2モード

### Feature flags
- `puyo-ai`の`nn`フィーチャーフラグでNN依存を制御。`puyo-wasm`は`nn`を有効にしてビルド。`nn`無しでは`nn_eval`、`mcts`、`inference_server`モジュールと`burn`依存がコンパイルから除外される。
- `puyo-ai`の`cuda`フィーチャーフラグ（`nn` + `burn/cuda-jit`）でGPU推論を有効化。
- `puyo-trainer`の`gpu`（デフォルト）/`cpu`フィーチャーフラグでバックエンドを切り替え。`gpu`は`burn/cuda-jit`+`burn/fusion`+`burn/autotune`を有効にする。

詳細は仕様書を参照: NN → `docs/spec/12-nn.md`、WASM → `docs/spec/10-wasm-bridge.md`、学習 → `docs/spec/13-trainer.md`、フロントエンド → `docs/spec/11-frontend.md`

## モデルファイル

- `artifacts/puyo_model.bin` (~500KB): 学習済みモデル（`BinFileRecorder`形式）
- `web/public/models/`: ブラウザ用デプロイ先

## ドキュメント更新ルール

**ソースファイルを変更したら、必ず対応する `docs/spec/` のドキュメントも更新すること。**

| 変更したファイル | 更新すべきドキュメント |
|----------------|----------------------|
| `crates/puyo-core/src/board.rs` | `docs/spec/02-board.md`, `docs/spec/01-architecture.md` |
| `crates/puyo-core/src/piece.rs` | `docs/spec/03-piece.md`, `docs/spec/01-architecture.md` |
| `crates/puyo-core/src/score.rs` | `docs/spec/05-score.md`, `docs/spec/01-architecture.md` |
| `crates/puyo-core/src/game.rs` | `docs/spec/06-game.md`, `docs/spec/01-architecture.md` |
| `crates/puyo-core/src/rng.rs` | `docs/spec/07-rng.md`, `docs/spec/01-architecture.md` |
| `crates/puyo-ai/src/eval.rs`, `crates/puyo-ai/src/nn_eval.rs` | `docs/spec/08-ai-eval.md` |
| `crates/puyo-ai/src/placement.rs`, `crates/puyo-ai/src/mcts.rs`, `crates/puyo-ai/src/inference_server.rs` | `docs/spec/09-ai-search.md` |
| `crates/puyo-wasm/src/lib.rs` | `docs/spec/10-wasm-bridge.md` |
| `web/src/*.ts` | `docs/spec/11-frontend.md` |
| `crates/puyo-nn/src/*.rs` | `docs/spec/12-nn.md` |
| `crates/puyo-trainer/src/**/*.rs` | `docs/spec/13-trainer.md` |

## 仕様書

詳細な仕様は `docs/spec/` 配下（00〜13の各ドキュメント）を参照。
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

依存方向: `puyo-core` ← `puyo-ai` ← `puyo-wasm`、`puyo-core` ← `puyo-nn` ← `puyo-trainer`

## アーキテクチャの要点

### Evaluator trait（多態性の中心）
`puyo-ai/src/eval.rs`の`Evaluator`トレイト（`find_best_move(&Board, &Piece, &Piece, &Piece) -> Option<(Placement, f64)>`）がAIの核。
- `SimulationEvaluator`: 仮想ぷよシミュレーションで盤面を評価（2手先読みBFS）
- `NnEvaluator`（`puyo-ai/src/nn_eval.rs`、`nn` feature flag有効時のみ）: Dual Head Network（Policy + Value）で評価。MCTSモード（`MctsConfig`付き、PUCT探索）とPolicy-onlyモード（WASM用、1回推論）の2モード

### Feature flag `nn`
`puyo-ai`の`nn`フィーチャーフラグでNN依存を制御。`puyo-wasm`は`nn`を有効にしてビルド。
`nn`無しでは`nn_eval`モジュールと`burn`依存がコンパイルから除外される。

### ボード表現とCNN
- ボード: 6列×14行、`PuyoColor` enum（Empty, Red, Green, Blue, Yellow）
- 盤面エンコーディング: one-hot 4ch + occupancy 1ch + adjacency 1ch = 6チャンネル × 14行 × 6列 = 504 floats → `[batch, 6, 14, 6]` テンソル
- コンテキストエンコーディング: 3ツモ × 2色 × 4 one-hot = 24 floats → `[batch, 24]` テンソル（`CONTEXT_TENSOR_SIZE = 24`）
- PuyoNet（Dual Head, FiLM Conditioning）: コンテキスト→FiLMジェネレータ(24→64→128→gamma[64]+beta[64]) → stem(6ch→64ch) → FiLMResidualBlock(64)×6(gamma*x+beta) → head_conv(64→128) → Pool([4,3]) → flatten(1536)
  - Policy Head: Linear(1536→256) → ReLU → Linear(256→24) → policy_logits
  - Value Head: Linear(1536→256) → ReLU → Linear(256→1) → value（tanhなし、累積割引報酬を出力）
- forward() 返り値: `(Tensor<B,2>, Tensor<B,2>)`（policy_logits, value）
- Burn 0.16、NdArrayバックエンド（CPU/WASM対応）

### WASMブリッジ
`WasmGame`構造体が`GameState`と`Box<dyn Evaluator>`を保持。
`load_nn_model()`でNNモデルをバイト列から読み込み（`BinBytesRecorder`使用）、`use_heuristic()`で`SimulationEvaluator`に切替。

### 学習パイプライン
- `generate-data`: SimulationEvaluator AIで~10Kゲーム → 盤面+コンテキスト(3ツモ)+選択手インデックスを記録（`data/training_data.bin`）
- `train`: 教師あり学習、Cross-Entropy損失（配置分類タスク）
- `self-play`: MCTSベースのAlphaZero self-playループ。ゲーム終了後に累積割引報酬(γ=0.99)を逆算してvalue_targetを計算。`AlphaZeroSample`（board_data, context_data, mcts_policy, value_target）を生成

### フロントエンド（web/src/）
- `main.ts`: エントリポイント、AI モード切替（heuristic/NN）
- `game-loop.ts`: ゲームループ管理（入力・描画・状態管理）
- `renderer.ts`: Canvas描画（目付きぷよ・ゴースト・落下アニメ）
- `model-loader.ts`: NNモデル読み込み（`web/public/models/`から）
- `wasm.ts` / `types.ts`: WASMモジュールローダーと型定義

## コーディングルール

### `&mut` の使用制限（Rust）
- `&mut self` は許可。
- それ以外の `&mut` パラメータは原則禁止。値の返却（タプル含む）で代替すること。
- 複数の可変状態を扱う場合は構造体にまとめ、メソッド（`&mut self`）で操作する。
- 例外: コンストラクタ内など `self` が存在しない文脈での `&mut`（例: `generate_piece(rng: &mut Rng)`）。

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
| `crates/puyo-ai/src/placement.rs`, `crates/puyo-ai/src/mcts.rs` | `docs/spec/09-ai-search.md` |
| `crates/puyo-wasm/src/lib.rs` | `docs/spec/10-wasm-bridge.md` |
| `web/src/*.ts` | `docs/spec/11-frontend.md` |
| `crates/puyo-nn/src/*.rs` | `docs/spec/12-nn.md` |
| `crates/puyo-trainer/src/**/*.rs` | `docs/spec/13-trainer.md` |

## 仕様書

詳細な仕様は `docs/spec/` 配下（00〜13の各ドキュメント）を参照。
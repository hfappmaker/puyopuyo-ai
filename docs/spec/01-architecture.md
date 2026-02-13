# システム構成

## 概要

Rust でゲームロジックとAIを実装し、WASM 経由でブラウザ上に表示する。

## 構成

| レイヤー | クレート/ディレクトリ | 役割 |
|----------|---------------------|------|
| puyo-core | `crates/puyo-core/` | 盤面（`Board`）・ぷよ組（`Piece`, `FallingPiece`）・連鎖（`chain`）・スコア（`score`）・ゲーム進行（`GameState`）・乱数（`Rng`） |
| puyo-ai | `crates/puyo-ai/` | 盤面評価（`eval`）・配置列挙（`placement`）・最善手探索（`search`） |
| puyo-nn | `crates/puyo-nn/` | CNN 価値ネットワーク（`PuyoValueNet`）・盤面テンソルエンコーディング（`encoding`） |
| puyo-trainer | `crates/puyo-trainer/` | 訓練データ生成（`generate-data`）・教師あり学習（`train`）・自己対戦強化学習（`self-play`） |
| puyo-wasm | `crates/puyo-wasm/` | `wasm-bindgen` による Rust ↔ JS ブリッジ（`WasmGame`） |
| web | `web/` | TypeScript + Vite によるフロントエンド（Canvas 描画・入力処理・ゲームループ・UI） |

## 依存関係

```
web (TypeScript)
  └── puyo-wasm (wasm-bindgen)
        ├── puyo-ai
        │     └── puyo-core
        └── puyo-core

puyo-trainer (バイナリ)
  ├── puyo-nn
  │     └── puyo-core
  ├── puyo-ai
  │     └── puyo-core
  └── puyo-core
```

## データフロー

1. プレイヤーがキー入力を行う（移動・回転・ドロップ・AI操作・リスタート）
2. フロントエンドが WASM ブリッジ経由で `GameState` を操作する
3. AI操作の場合、`search::find_best_move` が2手先読みで全配置を評価し最善手を返す（全配置がゲームオーバーとなる場合は1手先読みにフォールバックする）
4. ピース設置後、連鎖処理（`resolve_chains`）が自動実行される
5. フロントエンドが盤面・ネクスト・スコアを Canvas に描画する

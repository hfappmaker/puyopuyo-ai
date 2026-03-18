# ぷよぷよ AI

ぷよぷよの連鎖の組み方を学べる AI 学習ツール。
AI が算出した「次の一手」を見ながら、連鎖の構築パターンを学習できます。

## スクリーンショット

<!-- TODO: スクリーンショットを追加 -->
<!-- 例: ![ゲーム画面](docs/screenshot.png) -->

## 特徴

- Rust 実装の高速な連鎖シミュレーション
- AI（2手先読みシミュレーション評価 / CNN Dual Head + MCTS 評価）
- ブラウザ上で動作（WebAssembly）
- Canvas 描画（目付きぷよ・ゴースト表示・落下アニメーション）
- キーボード操作 + AI の一手実行 + リスタート

## 必要なもの

- [Rust](https://www.rust-lang.org/) (edition 2021)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/)
- [Node.js](https://nodejs.org/) + npm

## ビルド・起動

```bash
./scripts/build-wasm.sh   # WASM ビルド（テスト → コンパイル → npm install）
cd web && npm run dev      # 開発サーバー起動
```

その他のビルドコマンド（テスト、手動ビルド、学習パイプライン等）は [CLAUDE.md](CLAUDE.md) を参照。

## 操作方法

| キー | アクション |
|------|-----------|
| ← → | 左右移動 |
| ↑ | 回転 |
| ↓ | ソフトドロップ（底に着くと設置） |
| Space | AI 配置 / 確定 |
| R | リスタート |

## 仕様書

詳細な仕様は [docs/spec/](docs/spec/00-index.md) を参照してください。

## 技術スタック

- Rust (edition 2021) + wasm-bindgen
- Burn（ニューラルネットワークフレームワーク）
- TypeScript + Vite
- Canvas API
- wasm-pack

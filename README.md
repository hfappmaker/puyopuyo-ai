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
# WASM ビルド（テスト実行 → WASM コンパイル → npm install）
./scripts/build-wasm.sh

# 開発サーバー起動
cd web
npm run dev
```

### 手動でビルドする場合

```bash
# ※ ワークスペースルートから実行し、out-dir で web/wasm-pkg に出力する
wasm-pack build crates/puyo-wasm --target web --out-dir ../../web/wasm-pkg

# Web 依存インストール＆開発サーバー起動
cd web
npm install
npm run dev
```

> **注意**: Rust 側のコードを変更した場合は `wasm-pack build` を再実行し、
> Vite のキャッシュが残っている場合は `rm -rf node_modules/.vite` でクリアしてください。

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

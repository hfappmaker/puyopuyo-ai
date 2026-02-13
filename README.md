# ぷよぷよ AI

ぷよぷよの連鎖の組み方を学べる AI 学習ツール。
AI が算出した「次の一手」を見ながら、連鎖の構築パターンを学習できます。

## スクリーンショット

<!-- TODO: スクリーンショットを追加 -->
<!-- 例: ![ゲーム画面](docs/screenshot.png) -->

## 特徴

- Rust 実装の高速な連鎖シミュレーション
- 2 手先読み AI（8 項目の評価関数）
- ブラウザ上で動作（WebAssembly）
- Canvas 描画（目付きぷよ・ゴースト表示・落下アニメーション）
- キーボード操作 + AI の一手実行 + リスタート

## アーキテクチャ

```
┌─────────────────────────────────┐
│  Web Frontend (TypeScript)      │  Canvas 描画・入力・ゲームループ
├─────────────────────────────────┤
│  WASM Bridge (puyo-wasm)        │  Rust ↔ JS の橋渡し
├─────────────────────────────────┤
│  AI (puyo-ai)                   │  盤面評価・最善手探索
├─────────────────────────────────┤
│  Core (puyo-core)               │  盤面・ぷよ組・連鎖・スコア・RNG
└─────────────────────────────────┘
```

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
# ※ wasm-pack はワークスペースルートではなくクレートディレクトリから実行する
cd crates/puyo-wasm
wasm-pack build --target web --out-dir pkg

# Web 依存インストール＆開発サーバー起動
cd ../../web
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

## プロジェクト構成

```
puyopuyo-ai/
├── crates/
│   ├── puyo-core/     # 盤面・ぷよ組・連鎖・スコア・ゲーム進行
│   ├── puyo-ai/       # 評価関数・2手先読み探索
│   └── puyo-wasm/     # wasm-bindgen ブリッジ
├── web/
│   ├── src/           # TypeScript フロントエンド
│   └── index.html
├── docs/spec/         # 詳細仕様書
└── scripts/
    └── build-wasm.sh  # ビルドスクリプト
```

## AI の仕組み

1. 現在のぷよ組と次のぷよ組について、全配置パターンを列挙
2. 各配置後の盤面を 8 つの評価項目でスコアリング
   - 連鎖スコア、連鎖長、高さ、高さ均一性、連結度、潜在連鎖、中央寄り、ゲームオーバー回避
3. 最も評価の高い配置を「次の一手」として提示

詳細は [AI 評価関数](docs/spec/08-ai-eval.md)・[AI 探索](docs/spec/09-ai-search.md) を参照。

## 仕様書

詳細な仕様は [docs/spec/](docs/spec/00-index.md) を参照してください。

## 技術スタック

- Rust (edition 2021) + wasm-bindgen
- TypeScript + Vite
- Canvas API
- wasm-pack

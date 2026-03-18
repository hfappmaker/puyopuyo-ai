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

## アーキテクチャ

```
┌─────────────────────────────────┐
│  Web Frontend (TypeScript)      │  Canvas 描画・入力・ゲームループ
├─────────────────────────────────┤
│  WASM Bridge (puyo-wasm)        │  Rust ↔ JS の橋渡し
├─────────────────────────────────┤
│  AI (puyo-ai)                   │  盤面評価・最善手探索
├─────────────────────────────────┤
│  NN (puyo-nn)                   │  CNN Dual Head ネットワーク・エンコーディング
├─────────────────────────────────┤
│  Core (puyo-core)               │  盤面・ぷよ組・連鎖・スコア・RNG
└─────────────────────────────────┘

  puyo-trainer                       訓練パイプライン（データ生成・学習・自己対戦）
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

## プロジェクト構成

```
puyopuyo-ai/
├── crates/
│   ├── puyo-core/     # 盤面・ぷよ組・連鎖・スコア・ゲーム進行
│   ├── puyo-ai/       # 評価関数・最大3手先読み探索
│   ├── puyo-nn/       # CNN Dual Head ネットワーク
│   ├── puyo-trainer/  # 訓練パイプライン（データ生成・学習・自己対戦）
│   └── puyo-wasm/     # wasm-bindgen ブリッジ
├── web/
│   ├── src/           # TypeScript フロントエンド
│   └── index.html
├── docs/spec/         # 詳細仕様書
└── scripts/
    └── build-wasm.sh  # ビルドスクリプト
```

## AI の仕組み

1. 現在のぷよ組・次のぷよ組について、最大2手先の全配置パターンをBFS順で列挙
2. 各配置後の盤面を評価関数でスコアリング
   - **シミュレーション**: 仮想ぷよ（同色2個）を全合法配置に落として連鎖スコアの期待値を推定
   - **CNN**: ニューラルネットワークによる盤面評価（6ch エンコーディング）
3. 全深度を通じて最高評価を得た1手目の配置を「次の一手」として提示

詳細は [AI 評価関数](docs/spec/08-ai-eval.md)・[AI 探索](docs/spec/09-ai-search.md) を参照。

## 仕様書

詳細な仕様は [docs/spec/](docs/spec/00-index.md) を参照してください。

## 技術スタック

- Rust (edition 2021) + wasm-bindgen
- Burn（ニューラルネットワークフレームワーク）
- TypeScript + Vite
- Canvas API
- wasm-pack

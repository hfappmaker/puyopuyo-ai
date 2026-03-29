# システム構成

## 用語定義

| 用語 | 定義 |
|------|------|
| 軸ぷよ (axis) | ぷよ組の回転中心となるぷよ |
| 衛星ぷよ (satellite) | 軸ぷよの周りを回転するぷよ |
| ツモ (piece) | 落下してくる2個1組のぷよ |
| ネクスト | 次に落下するぷよ組 |
| ネクネク | ネクストの次に落下するぷよ組 |
| 連鎖 | 同色4個以上の消去が連続して発生する反応 |
| 致死列 | 2列目（列1）。この列の高さが可視行数（6）以上になるとゲームオーバー |
| 非可視行 | 行6〜7（7〜8行目）。連鎖判定の対象外。行6は重力の影響を受けるが、行7は重力の対象外 |
| ウォールキック | 壁際での回転時に自動的に位置を補正する仕組み |
| ゴースト | ハードドロップ先を示す半透明の表示 |
| ソフトドロップ | ↓キーで1マスずつ下降する操作 |
| ハードドロップ | 即座に着地位置まで落下し設置する操作 |
| AIプレビュー | Spaceキーで最善手の位置にピースを移動し、再度Spaceで設置する2段階操作 |

## 概要

Rust でゲームロジックとAIを実装し、WASM 経由でブラウザ上に表示する。

## 構成

| レイヤー | クレート/ディレクトリ | 役割 |
|----------|---------------------|------|
| game-core | `crates/game-core/` | ターン制ゲームの汎用抽象化（`Game` トレイト）。ゲーム非依存のAI探索を可能にする |
| puyo-core | `crates/puyo-core/` | ゲームパラメータ一元管理（`config`）・盤面（`Board`、連鎖解決を含む）・ぷよ組（`Piece`, `FallingPiece`）・スコア（`score`）・ゲーム進行（`GameState`）・乱数（`Rng`）・AI用ゲーム状態（`PuyoState`）・エンコーディング関数 |
| puyo-ai | `crates/puyo-ai/` | 盤面評価（`eval`）・配置列挙（`placement`）・ハッシュユーティリティ（`hash_util`）・`Game` トレイト実装（`puyo_game`: `PuyoGame`）・NN評価（`nn_eval`、`nn` feature）・MCTS探索（`mcts`、`nn` feature）・GPU推論サーバー（`inference_server`、`nn` feature） |
| puyo-nn | `crates/puyo-nn/` | CNN Dual Head ネットワーク（`PuyoNet`: Policy + Value）・盤面テンソルエンコーディング（`encoding`、`puyo-core` からの再エクスポートラッパー） |
| puyo-trainer | `crates/puyo-trainer/` | 訓練データ生成（`generate-data`）・教師あり学習（`train`）・自己対戦強化学習（`self-play`） |
| puyo-wasm | `crates/puyo-wasm/` | `wasm-bindgen` による Rust ↔ JS ブリッジ（`WasmGame`） |
| web | `web/` | TypeScript + Vite によるフロントエンド（Canvas 描画・入力処理・ゲームループ・UI） |

## 依存関係

```
web (TypeScript)
  └── puyo-wasm (wasm-bindgen)
        ├── puyo-ai
        │     ├── puyo-core
        │     │     └── game-core
        │     └── game-core
        └── puyo-core
              └── game-core

puyo-trainer (バイナリ)
  ├── puyo-nn
  │     └── puyo-core
  │           └── game-core
  ├── puyo-ai
  │     ├── puyo-core
  │     │     └── game-core
  │     └── game-core
  └── puyo-core
        └── game-core
```

`game-core` の `Game` トレイトがゲーム非依存のAI探索の中心的抽象化として機能し、`puyo-ai` の `Evaluator<G: Game>` や `MctsTree<G: Game>` が任意のターン制ゲームに対して汎用的に動作する。

## データフロー

1. プレイヤーがキー入力を行う（移動・回転・ドロップ・AI操作・リスタート）
2. フロントエンドが WASM ブリッジ経由で `GameState` を操作する
3. AI操作の場合、`Evaluator::find_best_move` が `G::State`（`PuyoState`）を受け取り、評価器に応じた探索で最善手を返す
4. ピース設置後、連鎖処理（`resolve_chains`）が自動実行される
5. フロントエンドが盤面・ネクスト・スコアを Canvas に描画する

## 学習パイプラインと成果物

`puyo-trainer` の3つのバイナリを順番に実行して NN モデルを生成する。
実行コマンド・パラメータ・成果物の詳細は [13-trainer.md](./13-trainer.md) を参照。

ブラウザで NN AI を使用する場合は学習済みモデルを `web/public/models/` にコピーして配置する。

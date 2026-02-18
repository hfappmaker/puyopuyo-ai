# Puyo Puyo AI

日本語でコミュニケーションすること。

## プロジェクト構造
- Rustワークスペース: `puyo-core`, `puyo-ai`, `puyo-nn`, `puyo-trainer`, `puyo-wasm`
- フロントエンド: TypeScript + Vite (`web/`)
- WASMブリッジでRust AIをブラウザに接続

## アーキテクチャ
- **Evaluator trait** (`puyo-ai/src/eval.rs`): ヒューリスティック/NN評価の多態性
- **Feature flag** `nn` (`puyo-ai`): NN依存を制御
- **Burn 0.16**: NNフレームワーク（NdArrayバックエンド、CPU/WASM対応）
- **BinFileRecorder**: モデル保存用 / **BinBytesRecorder**: WASMでのバイト読み込み用

## ボード表現
- One-hot: 5チャンネル (Empty, Red, Green, Blue, Yellow) × 13行 × 6列 = 390 floats
- CNN: Conv2d×3 → AdaptiveAvgPool2d → Linear×2 → スカラー値

## 学習パイプライン
- `generate-data`: ヒューリスティックAIで10Kゲーム → ~1Mサンプル (`data/training_data.bin`)
- `train`: 教師あり学習、MSE損失（連鎖数予測、z-scoreで正規化）
- `self-play`: TD(0) + ターゲットネットワークで強化学習
- 正規化パラメータ: `artifacts/norm_params.txt`
- 評価値 = 連鎖数（スコアや生存ではない）

## モデルファイル
- `artifacts/puyo_model.bin` (~500KB): 学習済みモデル
- `web/public/models/`: ブラウザ用デプロイ先

## ビルド・テスト
```bash
cargo build                    # Rustビルド
cargo test                     # テスト
bash scripts/build-wasm.sh     # WASMビルド
cd web && npm run dev          # フロントエンド開発サーバー
```

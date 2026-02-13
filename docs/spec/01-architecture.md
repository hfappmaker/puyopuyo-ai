# システムアーキテクチャ

| 項目 | 値 |
|------|-----|
| ステータス | 実装済み |
| クレート | 全体 |
| 最終更新 | 2026-02-13 |
| 対応ソース | `Cargo.toml`, `scripts/build-wasm.sh`, `web/vite.config.ts` |

## 構成概要

3つの Rust クレート + 1つの TypeScript フロントエンドで構成される。

```mermaid
graph TD
    subgraph Rust
        PC[puyo-core<br/>盤面・ルール・ゲーム進行]
        PA[puyo-ai<br/>評価関数・探索]
        PW[puyo-wasm<br/>WASM バインディング]
    end

    subgraph Web
        FE[web/<br/>TypeScript フロントエンド]
    end

    PC --> PA
    PC --> PW
    PA --> PW
    PW -->|wasm-bindgen| FE
```

### puyo-core (基盤クレート)

| モジュール | 責務 |
|-----------|------|
| `board` | 盤面データ構造、PuyoColor、重力、ゲームオーバー判定 |
| `piece` | ぷよ組、方向、配置、落下ピース、移動・回転 |
| `chain` | 連鎖検出（BFS flood-fill）、連鎖解決ループ |
| `score` | スコア計算（ボーナステーブル） |
| `game` | ゲーム状態管理、フェーズ遷移、tick |
| `rng` | Xorshift64 乱数生成 |

外部依存: なし（`std` のみ）

### puyo-ai (AI クレート)

| モジュール | 責務 |
|-----------|------|
| `eval` | 盤面評価関数（8項目の重み付き評価） |
| `search` | 深さ1/2探索、最善手探索 |
| `placement` | 合法配置列挙、同色重複排除 |

依存: `puyo-core`

### puyo-wasm (WASM ブリッジクレート)

| モジュール | 責務 |
|-----------|------|
| `lib` | WasmGame ラッパー、JS バインディング |

依存: `puyo-core`, `puyo-ai`, `wasm-bindgen`

クレートタイプ: `cdylib`（動的ライブラリ）+ `rlib`（テスト用）

### web (フロントエンド)

| ファイル | 責務 |
|---------|------|
| `main.ts` | エントリポイント、初期化 |
| `wasm.ts` | WASM モジュールの動的ロード |
| `types.ts` | WasmGame の TypeScript 型定義 |
| `constants.ts` | 共有定数（色、サイズ、速度） |
| `game-loop.ts` | ゲームループ（60fps）、入力ディスパッチ |
| `input.ts` | キーボード入力ハンドラ |
| `renderer.ts` | Canvas 2D レンダリング |
| `ui.ts` | DOM ベースの UI 更新 |

依存: Vite, `vite-plugin-wasm`, `vite-plugin-top-level-await`

## データフロー

```mermaid
sequenceDiagram
    participant User as ユーザ入力
    participant Input as InputHandler
    participant GL as GameLoop
    participant WASM as WasmGame
    participant GS as GameState (Rust)
    participant R as Renderer
    participant UI as UI

    User->>Input: keydown/keyup
    Input->>GL: consumeJustPressed()
    GL->>WASM: move_left() / hard_drop() / tick() 等
    WASM->>GS: 対応する GameState メソッド
    GS-->>WASM: 結果
    WASM-->>GL: bool / u32
    GL->>WASM: get_board() / get_current_piece()
    WASM->>GS: Board::to_flat() 等
    GS-->>WASM: Vec<u8>
    WASM-->>GL: Uint8Array
    GL->>R: render(game)
    GL->>UI: update(game)
    R-->>User: Canvas 描画
    UI-->>User: DOM 更新
```

### AI モードのデータフロー

```mermaid
sequenceDiagram
    participant GL as GameLoop
    participant WASM as WasmGame
    participant AI as search::find_best_move
    participant Eval as eval::evaluate

    GL->>WASM: ai_play_move()
    WASM->>AI: find_best_move(board, current, next)
    AI->>AI: enumerate_placements()
    loop 各配置
        AI->>AI: simulate_placement()
        AI->>Eval: evaluate(board)
        Eval-->>AI: f64
    end
    AI-->>WASM: SearchResult
    WASM->>WASM: apply_placement()
    WASM-->>GL: chain_count
```

## ビルドパイプライン

`scripts/build-wasm.sh` による統一ビルド:

```mermaid
flowchart LR
    A[cargo test --workspace] --> B[wasm-pack build]
    B --> C[npm install]
    C --> D[npm run dev]
```

### 手順

1. **テスト実行**: `cargo test --workspace` — 全クレートのテストを実行
2. **WASM ビルド**: `wasm-pack build crates/puyo-wasm --target web --out-dir web/wasm-pkg`
   - `--target web`: ESM モジュールとして生成
   - `--out-dir`: web ディレクトリ直下に出力
3. **依存インストール**: `cd web && npm install`
4. **開発サーバ**: `npm run dev`（Vite 開発サーバ）

### Vite 設定

```typescript
// web/vite.config.ts
export default defineConfig({
  plugins: [wasm(), topLevelAwait()],
  build: { target: "esnext" },
});
```

- `vite-plugin-wasm`: WASM ファイルの ESM インポートをサポート
- `vite-plugin-top-level-await`: トップレベル await のサポート
- `target: "esnext"`: 最新 JS 機能を使用（WASM 対応ブラウザを前提）

## ワークスペース構成

```toml
# Cargo.toml (ルート)
[workspace]
members = [
    "crates/puyo-core",
    "crates/puyo-ai",
    "crates/puyo-wasm",
]
resolver = "2"
```

Cargo ワークスペースにより3クレートを統合管理。

# WASM ブリッジ API 仕様

| 項目 | 値 |
|------|-----|
| ステータス | 実装済み |
| クレート | puyo-wasm |
| 最終更新 | 2026-02-13 |
| 対応ソース | `crates/puyo-wasm/src/lib.rs`, `web/src/types.ts` |

## 概要

Rust のゲームロジックを WebAssembly 経由でフロントエンドに公開する。`wasm-bindgen` を使用して JS バインディングを生成する。

## WasmGame 構造体

```rust
#[wasm_bindgen]
pub struct WasmGame {
    state: GameState,
}
```

内部で `GameState` をラップし、WASM 互換のインターフェースを提供する。

## API 一覧

### コンストラクタ

| メソッド | シグネチャ | 説明 |
|----------|-----------|------|
| `new` | `fn new(seed: u64) -> WasmGame` | 指定シードでゲームを初期化 |

JS 側からは `new WasmGame(BigInt(seed))` で呼び出す。

### 状態取得

| メソッド | 戻り値 | 説明 |
|----------|--------|------|
| `get_board()` | `Vec<u8>` | 盤面データ（84バイト） |
| `get_current_piece()` | `Vec<u8>` | 現在ピース情報（6バイト or 空） |
| `get_next_piece()` | `Vec<u8>` | ネクストピース情報（2バイト） |
| `get_score()` | `u32` | 現在スコア |
| `get_max_chain()` | `u32` | 最大連鎖数 |
| `get_phase()` | `u8` | ゲームフェーズ |
| `get_total_pieces()` | `u32` | 設置済みピース数 |

### 操作

| メソッド | 戻り値 | 説明 |
|----------|--------|------|
| `move_left()` | `bool` | 左移動。成功なら `true` |
| `move_right()` | `bool` | 右移動。成功なら `true` |
| `rotate_cw()` | `bool` | 時計回り回転。成功なら `true` |
| `rotate_ccw()` | `bool` | 反時計回り回転。成功なら `true` |
| `hard_drop()` | `u32` | ハードドロップ。連鎖数を返す（0=連鎖なし） |
| `tick(gravity: f32)` | `u32` | 重力更新。着地時の連鎖数を返す |

### AI

| メソッド | 戻り値 | 説明 |
|----------|--------|------|
| `ai_best_move()` | `Vec<u8>` | 最善手を計算（2バイト or 空） |
| `ai_play_move()` | `u32` | 最善手を計算して即時適用。連鎖数を返す |

### 制御

| メソッド | 戻り値 | 説明 |
|----------|--------|------|
| `restart(seed: u64)` | `void` | ゲームをリスタート |

## データエンコーディング

### 盤面データ (get_board)

84バイトの `Uint8Array`。列優先・下から上の順。

```
[col0_row0, col0_row1, ..., col0_row13, col1_row0, ..., col5_row13]
```

各バイトは `PuyoColor` の `u8` 値:
- 0: Empty
- 1: Red
- 2: Green
- 3: Blue
- 4: Yellow

アクセス: `board[col * 14 + row]`

### 現在ピースデータ (get_current_piece)

6バイトの `Uint8Array`。ピースがない場合は空配列。

| インデックス | 内容 | 型 |
|------------|------|-----|
| 0 | axis_color | u8 (PuyoColor) |
| 1 | satellite_color | u8 (PuyoColor) |
| 2 | col | u8 (列番号 0-5) |
| 3 | row_int | u8 (行の整数部) |
| 4 | row_frac_x100 | u8 (行の小数部 × 100) |
| 5 | orientation | u8 (0-3) |

行の復元: `row = row_int + row_frac_x100 / 100`

### ネクストピースデータ (get_next_piece)

2バイトの `Uint8Array`。

| インデックス | 内容 |
|------------|------|
| 0 | axis_color |
| 1 | satellite_color |

### フェーズエンコーディング (get_phase)

| 値 | フェーズ |
|----|---------|
| 0 | Falling |
| 1 | Resolving |
| 2 | GameOver |

### AI 最善手データ (ai_best_move)

2バイトの `Uint8Array`。手が見つからない場合は空配列。

| インデックス | 内容 |
|------------|------|
| 0 | col (軸列 0-5) |
| 1 | orientation (0=N, 1=E, 2=S, 3=W) |

## TypeScript 型定義

```typescript
interface WasmGame {
  get_board(): Uint8Array;
  get_current_piece(): Uint8Array;
  get_next_piece(): Uint8Array;
  get_score(): number;
  get_max_chain(): number;
  get_phase(): number;
  get_total_pieces(): number;
  move_left(): boolean;
  move_right(): boolean;
  rotate_cw(): boolean;
  rotate_ccw(): boolean;
  hard_drop(): number;
  tick(gravity: number): number;
  ai_best_move(): Uint8Array;
  ai_play_move(): number;
  restart(seed: bigint): void;
  free(): void;
}
```

`free()` は wasm-bindgen が自動生成するメモリ解放メソッド。

## WASM モジュールのロード

```typescript
// web/src/wasm.ts
const mod = await import("../../crates/puyo-wasm/pkg/puyo_wasm.js");
await mod.default();  // WASM 初期化
```

`vite-plugin-wasm` と `vite-plugin-top-level-await` により、ESM モジュールとしてインポート可能。

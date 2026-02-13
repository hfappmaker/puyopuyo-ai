# 盤面仕様

| 項目 | 値 |
|------|-----|
| ステータス | 実装済み |
| クレート | puyo-core |
| 最終更新 | 2026-02-13 |
| 対応ソース | `crates/puyo-core/src/board.rs` |

## 定数

| 定数名 | 値 | 説明 |
|--------|-----|------|
| `COLS` | 6 | 列数 |
| `ROWS` | 14 | 行数（可視12行 + 非可視2行） |
| `VISIBLE_ROWS` | 12 | 可視行数 |

### 座標系

- 列: 0（左端）〜 5（右端）
- 行: 0（最下段）〜 13（最上段）
- 行0〜11が可視領域、行12〜13が非可視領域（出現用）

## PuyoColor 列挙型

`#[repr(u8)]` で定義され、u8値との相互変換が可能。

| バリアント | 値 | 説明 |
|-----------|-----|------|
| `Empty` | 0 | 空セル |
| `Red` | 1 | 赤 |
| `Green` | 2 | 緑 |
| `Blue` | 3 | 青 |
| `Yellow` | 4 | 黄 |

### メソッド

| メソッド | シグネチャ | 説明 |
|----------|-----------|------|
| `from_u8` | `fn from_u8(v: u8) -> Self` | u8 から変換。未知の値は `Empty` |
| `is_color` | `fn is_color(self) -> bool` | `Empty` でなければ `true` |

## Board 構造体

```rust
pub struct Board {
    pub columns: [[PuyoColor; ROWS]; COLS],
}
```

列優先（column-major）配列。`columns[col][row]` でアクセス。

### API

| メソッド | シグネチャ | 説明 |
|----------|-----------|------|
| `new` | `fn new() -> Self` | 全セル `Empty` の空盤面を生成 |
| `column_height` | `fn column_height(&self, col: usize) -> usize` | 指定列の高さ（最上段の非空セルの行+1） |
| `get` | `fn get(&self, col: usize, row: usize) -> PuyoColor` | 指定座標の色を取得 |
| `set` | `fn set(&mut self, col: usize, row: usize, color: PuyoColor)` | 指定座標に色を設定 |
| `drop_puyo` | `fn drop_puyo(&mut self, col: usize, color: PuyoColor) -> Option<usize>` | 指定列にぷよを落下。着地行を返す。列が満杯なら `None` |
| `apply_gravity` | `fn apply_gravity(&mut self)` | 全列で空隙を詰めてぷよを落下させる |
| `is_game_over` | `fn is_game_over(&self) -> bool` | ゲームオーバー判定 |
| `to_flat` | `fn to_flat(&self) -> Vec<u8>` | 列優先でフラット化した u8 配列（WASM転送用） |

### ゲームオーバー条件

```
column_height(2) > VISIBLE_ROWS  // 3列目の高さが12を超えたら
```

3列目（列インデックス2）の高さが `VISIBLE_ROWS`(12) を超えた場合にゲームオーバーとなる。

### apply_gravity アルゴリズム

各列について、read/write ポインタ方式で空隙を詰める:

1. `write = 0` から開始
2. `read` を 0 から ROWS-1 まで走査
3. `columns[col][read]` が色ぷよなら `columns[col][write]` にコピーし `write++`
4. コピー元と先が異なる場合、元を `Empty` にクリア

### to_flat エンコーディング

列優先・下から上の順で `u8` に変換:

```
[col0_row0, col0_row1, ..., col0_row13, col1_row0, ..., col5_row13]
```

合計 84 バイト（6列 × 14行）。

## テスト要件

| テスト | 検証内容 |
|--------|---------|
| `test_new_board_is_empty` | 新規盤面の全列高さが0 |
| `test_drop_puyo` | ぷよ落下時の着地行と列高さ |
| `test_column_full` | 列満杯時に `None` が返る |
| `test_apply_gravity` | 空隙がある盤面で重力適用後にぷよが正しく落下 |
| `test_game_over` | 3列目が12行を超えるとゲームオーバー |
| `test_to_flat` | フラット化配列の長さと全値が0 |

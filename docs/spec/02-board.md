# 盤面仕様

## サイズ

- 6列 × 14行（可視12行 + 非可視2行）
- 列: 0（左端）〜 5（右端）
- 行: 0（最下段）〜 13（最上段・非可視行）

## データ構造

列優先（column-major）の2次元配列 `columns[col][row]` で格納する。

```rust
pub struct Board {
    pub columns: [[PuyoColor; 14]; 6],
}
```

## ぷよの色

`PuyoColor` 列挙型で表現する。

| 値 | 色 | repr |
|----|----|------|
| Empty | 空（何もないセル） | 0 |
| Red | 赤 | 1 |
| Green | 緑 | 2 |
| Blue | 青 | 3 |
| Yellow | 黄 | 4 |

## 定数

| 定数 | 値 | 説明 |
|------|-----|------|
| `COLS` | 6 | 列数 |
| `ROWS` | 14 | 行数（可視12 + 非可視2） |
| `VISIBLE_ROWS` | 12 | 可視行数 |
| `SPAWN_COL` | 2 | スポーン列（3列目） |

## 主な操作

| 操作 | 説明 |
|------|------|
| `drop_puyo(col, color)` | 指定列にぷよを落とし、積まれた行を返す。列が満杯なら `assert` でパニック |
| `apply_gravity()` | 行0〜12のぷよを落下させて空隙を埋める。行13（最上非可視行）は対象外 |
| `column_height(col)` | 指定列の高さ（底からの連続した非空セル数）をボトムアップ走査で返す。行13に孤立ぷよがあっても無視される |
| `has_isolated_top_puyo(col)` | row 13にぷよがあり、row 12が空の場合に `true` を返す。連鎖消去後に孤立したぷよの検出に使用 |
| `is_game_over()` | `SPAWN_COL`（列2）の高さが `VISIBLE_ROWS`（12）以上なら `true` |
| `to_flat()` | WASM転送用に列優先・下から上の `Vec<u8>` に変換する（長さ84）。イテレータチェイン(`flat_map`)で実装 |

## ゲームオーバー判定

`SPAWN_COL`（列2）の `column_height` が `VISIBLE_ROWS`（12）以上になるとゲームオーバーとなる（`>=` 判定）。
つまり高さ12＝行11まで埋まった時点でゲームオーバー。可視行（行0〜11）が全て埋まると発動する。

## 非可視行の挙動

- **行12（13段目）**: 連鎖計算（`find_connected_groups`）の対象外。ここにぷよが存在するとゲームオーバー判定が発生する。
- **行13（14段目）**: 連鎖計算・重力（`apply_gravity`）の対象外。ここにぷよを置いてもゲームオーバーにはならないが、重力でフィールドに降りてくることもない（ゲームが終わるまで引っかかったまま）。`column_height` はボトムアップ走査のため、行13に孤立ぷよがあっても高さに影響しない。

## 連鎖解決メソッド

連鎖ロジックは `Board` の impl メソッドとして統合されている（以前は `chain.rs` に分離していたが、`board.rs` に統合された）。

### 定数

| 定数 | 値 | 説明 |
|------|-----|------|
| `MIN_GROUP_SIZE` | 4 | 消去に必要な最小グループサイズ |

### データ構造

```rust
pub struct Group {
    pub color: PuyoColor,
    pub cells: Vec<(usize, usize)>,  // (col, row)
}

pub struct ChainResult {
    pub chain_count: u32,        // 連鎖数
    pub score: u32,              // 合計スコア
}
```

### メソッド

| メソッド | 説明 |
|---------|------|
| `find_connected_groups()` | BFS（幅優先探索）で可視行（行0〜行11）の同色連結グループを全て検出する。グループサイズの制限なし |
| `find_clearable_groups()` | `find_connected_groups()` のうち `MIN_GROUP_SIZE`（4）個以上のグループのみ返す |
| `resolve_chains()` | 内部で1ステップずつ連鎖を処理（消去→スコア計算→重力適用）し、全連鎖を解決して `ChainResult`（連鎖数・合計スコア）を返す |

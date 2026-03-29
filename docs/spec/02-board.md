# 盤面仕様

## サイズ

- 3列 × 8行（可視6行 + 非可視2行）
- 列: 0（左端）〜 2（右端）
- 行: 0（最下段）〜 7（最上段・非可視行）

## データ構造

列優先（column-major）の2次元配列 `columns[col][row]` で格納する。

```rust
pub struct Board {
    pub columns: [[PuyoColor; 8]; 3],
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

Yellow は列挙型に存在するが、`NUM_COLORS=3` の場合は使用されない。

### メソッド

| メソッド | 説明 |
|---------|------|
| `PuyoColor::from_u8(v)` | `u8` から `PuyoColor` に変換する。1=Red, 2=Green, 3=Blue, 4=Yellow、それ以外は `Empty` |
| `PuyoColor::all_colors()` | アクティブな色のスライスを返す。`&ALL_COLOR_VARIANTS[..NUM_COLORS]`（NUM_COLORS=3 なら Red, Green, Blue） |
| `is_color(self)` | `Empty` でなければ `true` を返す |

## 定数

| 定数 | 値 | 説明 |
|------|-----|------|
| `COLS` | 3 | 列数 |
| `ROWS` | 8 | 行数（可視6 + 非可視2） |
| `VISIBLE_ROWS` | 6 | 可視行数 |
| `SPAWN_COL` | 1 | スポーン列（2列目） |
| `NUM_COLORS` | 3 | アクティブな色数（Red, Green, Blue）。PuyoColor 列挙型は Yellow=4 も含むが、NUM_COLORS=3 の場合は使用されない |

これらの定数は `config.rs`（`puyo_core::config`）で一元管理されており、`board.rs` から再エクスポートされている。`config.rs` の値を変更すると、依存する全クレート（`puyo-player`, `puyo-nn`, `puyo-trainer`）の関連定数（`NUM_CHANNELS`, `TENSOR_SIZE`, `NUM_ACTIONS` 等）が自動的に伝播する。唯一の例外は `model.rs` の `POOL_H` / `POOL_W` で、これらはアーキテクチャハイパーパラメータとして手動調整が必要。定数変更後は再コンパイルと NN モデルの再学習が必要。

## 主な操作

| 操作 | 説明 |
|------|------|
| `new()` | 全セルが `Empty` の空盤面を生成する。`Default` トレイトも実装されている |
| `get(col, row)` | 指定セルの `PuyoColor` を返す |
| `set(col, row, color)` | 指定セルに `PuyoColor` を設定する |
| `drop_puyo(col, color)` | 指定列にぷよを落とし、積まれた行を返す。列が満杯なら `assert` でパニック |
| `apply_gravity()` | 行0〜6のぷよを落下させて空隙を埋める。行7（最上非可視行）は対象外 |
| `column_height(col)` | 指定列の高さ（底からの連続した非空セル数）をボトムアップ走査で返す。行7に孤立ぷよがあっても無視される |
| `column_info(col)` | `(usize, bool)` を返す。第1要素は列の高さ（`column_height` と同値）、第2要素は row 7 に孤立ぷよがあるか（row 7 にぷよがあり row 6 が空なら `true`）。`column_height` は内部で `column_info(col).0` を呼ぶ |
| `is_game_over()` | `SPAWN_COL`（列1）の高さが `VISIBLE_ROWS`（6）以上なら `true` |
| `to_flat()` | WASM転送用に列優先・下から上の `Vec<u8>` に変換する（長さ24）。イテレータチェイン(`flat_map`)で実装 |

## ゲームオーバー判定

`SPAWN_COL`（列1）の `column_height` が `VISIBLE_ROWS`（6）以上になるとゲームオーバーとなる（`>=` 判定）。
つまり高さ6＝行5まで埋まった時点でゲームオーバー。可視行（行0〜5）が全て埋まると発動する。

## 非可視行の挙動

- **行6（7段目）**: 連鎖計算（`find_connected_groups`）の対象外。ここにぷよが存在するとゲームオーバー判定が発生する。
- **行7（8段目）**: 連鎖計算・重力（`apply_gravity`）の対象外。ここにぷよを置いてもゲームオーバーにはならないが、重力でフィールドに降りてくることもない（ゲームが終わるまで引っかかったまま）。`column_height` はボトムアップ走査のため、行7に孤立ぷよがあっても高さに影響しない。

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
| `find_connected_groups()` | BFS（幅優先探索）で可視行（行0〜行5）の同色連結グループを全て検出する。グループサイズの制限なし |
| `find_clearable_groups()` | `find_connected_groups()` のうち `MIN_GROUP_SIZE`（4）個以上のグループのみ返す |
| `resolve_chains()` | 内部で1ステップずつ連鎖を処理（消去→スコア計算→重力適用）し、全連鎖を解決して `ChainResult`（連鎖数・合計スコア）を返す |

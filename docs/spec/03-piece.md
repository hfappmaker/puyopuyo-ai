# ぷよ組・ツモ仕様

## ぷよ組の構成

2個1組のぷよ（軸ぷよ + 衛星ぷよ）で構成される。

```rust
pub struct Piece {
    pub axis_color: PuyoColor,
    pub satellite_color: PuyoColor,
}
```

## 方向 (Orientation)

衛星ぷよは軸ぷよに対して4方向に配置できる。

| 方向 | 衛星の位置 | オフセット (dcol, drow) |
|------|-----------|----------------------|
| North | 軸の上 | (0, +1) |
| East | 軸の右 | (+1, 0) |
| South | 軸の下 | (0, -1) |
| West | 軸の左 | (-1, 0) |

## 回転

- 時計回り (CW): North → East → South → West → North
- 反時計回り (CCW): North → West → South → East → North

## 配置 (Placement)

AIの探索結果は「軸ぷよをどの列に、どの方向で置くか」で表現される。

```rust
pub struct Placement {
    pub col: usize,            // 軸の列 (0-5)
    pub orientation: Orientation,
}
```

## 落下中のぷよ組 (FallingPiece)

ゲームプレイ中のぷよ組の状態を管理する。

```rust
pub struct FallingPiece {
    pub piece: Piece,
    pub col: usize,          // 軸の列
    pub row: f32,            // 軸の行（小数で滑らかな落下を表現）
    pub orientation: Orientation,
}
```

## 方向のシリアライズ

`Orientation::as_u8()` メソッドで整数に変換できる（0=North, 1=East, 2=South, 3=West）。WASM境界やレンダリング情報の受け渡しで使用。

## 出現位置

- `SPAWN_COL`（列2）の `VISIBLE_ROWS as f32`（12.0）に出現
- 初期方向は North（衛星が上）

## ウォールキック

壁際で回転できない場合、衛星の方向と反対に1マスずらして回転を成立させる。

1. 回転後の方向で配置可能か判定
2. 配置不可の場合、`kick_col = col - dc`（衛星の反対方向）に移動して再判定
3. それでも不可なら回転失敗

## 操作

| 操作 | 説明 |
|------|------|
| `try_move_left(&Board)` | 左に1マス移動（境界・衝突チェック付き） |
| `try_move_right(&Board)` | 右に1マス移動（境界・衝突チェック付き） |
| `try_rotate_cw(&Board)` | 時計回り回転（ウォールキック付き） |
| `try_rotate_ccw(&Board)` | 反時計回り回転（ウォールキック付き） |

## 衝突判定 (`can_occupy`)

`FallingPiece` の内部メソッド。軸と衛星の両方について以下の2段階チェックを行う:

1. **高さベースチェック**: 対象セルの行が列の高さ（`column_height`）未満なら衝突
2. **セルレベルチェック**: 対象セルが既にぷよで占有されていれば衝突（row 13 の孤立ぷよ対策）

セルレベルチェックは、`column_height` がボトムアップ走査で孤立ぷよを無視するため、高さベースチェックだけでは検出できない row 13 の孤立ぷよとの重複を防ぐ。

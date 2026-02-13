# 連鎖処理仕様

## 連鎖の仕組み

1. 同色ぷよが4個以上つながっていると消去される
2. 消去後、上にあったぷよが重力で落下する
3. 落下の結果、再び4個以上つながれば連続して消去される
4. これ以上消えなくなるまで繰り返す

## 連鎖判定の範囲

- 可視行（行0〜行11）のみが連鎖判定の対象
- 非可視行（行12）にあるぷよは連鎖判定に含まれない
- 非可視行のぷよは重力による落下の影響は受ける（消去後に落ちてくる）

## グループ検出アルゴリズム

BFS（幅優先探索）によるフラッドフィルで同色の連結グループを検出する。

```rust
pub struct Group {
    pub color: PuyoColor,
    pub cells: Vec<(usize, usize)>,  // (col, row)
}
```

4方向（上下左右）の隣接セルを探索し、4個以上のグループを返す。

## 連鎖結果

```rust
pub struct ChainResult {
    pub chain_count: u32,        // 連鎖数
    pub score: u32,              // 合計スコア
    pub steps: Vec<ChainStep>,   // 各ステップの詳細
}

pub struct ChainStep {
    pub chain_num: u32,          // 連鎖番号（1始まり）
    pub groups: Vec<Group>,      // 消去されたグループ
    pub score: u32,              // このステップのスコア
}
```

## 連鎖の例（2連鎖）

```
初期状態:         1連鎖後（赤消去）:  重力適用後:      2連鎖（青消去）:
Col 0: B B B      Col 0: B B B       Col 0: B B B B   Col 0: (empty)
Col 1: R R R R B  Col 1: B           Col 1: B         Col 1: (empty)
```

1. 赤4個が消える（1連鎖目）
2. 残った青が落下して青3個と合流（計4個）
3. 青4個が消える（2連鎖目）

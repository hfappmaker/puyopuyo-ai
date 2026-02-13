# 乱数生成仕様

| 項目 | 値 |
|------|-----|
| ステータス | 実装済み |
| クレート | puyo-core |
| 最終更新 | 2026-02-13 |
| 対応ソース | `crates/puyo-core/src/rng.rs` |

## 概要

決定論的な擬似乱数生成器。同じシードからは常に同じ乱数列が生成されるため、ゲームの再現性を保証する。

## Rng 構造体

```rust
pub struct Rng {
    state: u64,
}
```

内部状態は 64bit 整数1個のみ。

## アルゴリズム: Xorshift64

```rust
pub fn next_u64(&mut self) -> u64 {
    let mut x = self.state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    self.state = x;
    x
}
```

Marsaglia (2003) の Xorshift アルゴリズムの変種。シフト定数は (13, 7, 17)。

### 性質

- 周期: 2^64 - 1（state=0 を除く全値を巡回）
- 速度: 高速（XOR とシフトのみ）
- 品質: 統計テストにおける品質は中程度（ゲーム用途では十分）

## 初期化

```rust
pub fn new(seed: u64) -> Self {
    Rng {
        state: if seed == 0 { 1 } else { seed },
    }
}
```

- `seed == 0` の場合、`state = 1` に補正
- Xorshift は state=0 が吸収状態（永遠に0を出力）のため、これを回避する

## 範囲指定乱数

```rust
pub fn next_range(&mut self, n: u32) -> u32 {
    (self.next_u64() % n as u64) as u32
}
```

`[0, n)` の範囲の値を返す。

### 既知の制限: modulo バイアス

`u64 % n` は `n` が 2^64 の約数でない限り均等分布にならない（modulo バイアス）。ただし、`n` が小さい値（例: 4）で `u64` の範囲が十分大きいため、実用上のバイアスは無視できるレベル。

## 使用箇所

- `GameState::generate_piece`: 色の選択に `rng.next_range(4)` を使用
  - 軸ぷよ: `rng.next_range(4) + 1` → PuyoColor 1〜4
  - 衛星ぷよ: `rng.next_range(4) + 1` → PuyoColor 1〜4

## テスト要件

| テスト | 検証内容 |
|--------|---------|
| `test_deterministic` | 同じシードから同じ乱数列が生成される |
| `test_range` | `next_range(4)` の出力が常に 0〜3 の範囲内 |
| `test_zero_seed_handled` | seed=0 でも正常動作（0以外の値を出力） |

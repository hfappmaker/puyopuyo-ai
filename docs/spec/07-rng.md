# 乱数生成仕様

## 概要

ぷよ組の色をランダムに決定するための乱数生成モジュール。毎回異なるゲーム展開となる。

## 関数

| 関数 | 説明 |
|------|------|
| `time_seed() -> u64` | 現在時刻からランダムなシードを生成する。内部で `splitmix64` を使用してビットを分散させる |
| `random_piece() -> Piece` | `time_seed()` を使ってランダムなピースを生成する |

## time_seed

プラットフォームに応じた現在時刻を取得し、`splitmix64` finalizer でハッシュ化する。

- **ネイティブ**: `SystemTime::now()` のナノ秒を使用
- **WASM**: `js_sys::Date::now() * 1_000_000.0` をマイクロ秒精度で使用

## random_piece

`time_seed()` で得たシードから2色を決定し、`Piece::new(axis, satellite)` を生成する。

1. `x = time_seed()`
2. 軸色: `(x % NUM_COLORS) + 1` で色番号を決定
3. シードを追加ミキシング（`x ^= x >> 30; x *= 0x517cc1b727220a95; x ^= x >> 27`）
4. 衛星色: `(x % NUM_COLORS) + 1` で色番号を決定

NUM_COLORS=3 の場合、色番号 1〜3（Red, Green, Blue）が生成される。

## splitmix64

内部ヘルパー関数。`u64` のシード値を分散の良いハッシュに変換する finalizer。

```
s = (s ^ (s >> 30)) × 0xBF58476D1CE4E5B9
s = (s ^ (s >> 27)) × 0x94D049BB133111EB
s = s ^ (s >> 31)
```

## 用途

- `GameState::new()` で3つのぷよ組（現在・ネクスト・ネクネク）の生成に使用
- `Game::advance_turn()` で MCTS 探索中のランダムツモ生成に使用
- 各呼び出しごとに `time_seed()` が新しいシードを生成するため、決定論的な再現性はない（ゲーム再現性が必要な場合は別の仕組みが必要）

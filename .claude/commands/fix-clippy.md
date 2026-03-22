ワークスペース全体の clippy 警告をゼロにしてください。

## 手順

1. `cargo clippy --fix --workspace --allow-dirty` を実行し、自動修正可能な警告を先に修正する
2. `cargo clippy --workspace` を実行し、残った警告を確認する
3. 残った警告があれば、コードを手動で修正する（各警告の内容に応じて適切に対処）
4. `cargo clippy --workspace` で警告ゼロを確認する
5. `cargo test --workspace` でテストが全パスすることを確認する
6. 修正内容の一覧を最後に報告する

## 注意事項

- `#[allow(clippy::...)]` による抑制は、構造的に修正が困難な場合（`too_many_arguments` 等）のみ使用する
- テストコードも含めて全て修正する
- 修正によって既存の動作が変わらないことをテストで確認する

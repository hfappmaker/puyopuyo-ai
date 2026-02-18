#!/bin/bash
# Claude Codeプラグインの自動インストール
# devcontainer再ビルド時にプラグインキャッシュが消えるため、自動で再インストールする

# Claude Codeがなければスキップ
command -v claude >/dev/null 2>&1 || exit 0

# マーケットプレイス追加
claude plugin marketplace add anthropics/claude-plugins-official 2>/dev/null || true

# プラグインインストール
plugins=(
  commit-commands
  context7
  code-review
  code-simplifier
  frontend-design
  security-guidance
  pr-review-toolkit
  claude-md-management
  github
  rust-analyzer-lsp
  typescript-lsp
)
for p in "${plugins[@]}"; do
  claude plugin install "${p}@claude-plugins-official" --scope project || true
done

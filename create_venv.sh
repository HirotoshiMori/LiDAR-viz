#!/usr/bin/env bash
set -e

# 今のプロジェクト情報
PROJECT_DIR="$(pwd)"
PROJECT_NAME="$(basename "$PROJECT_DIR")"

BASE_DIR="$HOME/local_envs/$PROJECT_NAME"
VENV_DIR="$BASE_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

VSCODE_DIR="$PROJECT_DIR/.vscode"
SETTINGS_FILE="$VSCODE_DIR/settings.json"

echo "==============================================="
echo ">>> プロジェクト名: $PROJECT_NAME"
echo ">>> 仮想環境ディレクトリ: $VENV_DIR"
echo ">>> Python 実行ファイル: $PYTHON_BIN"
echo "==============================================="

echo ">>> [1/4] local_envs フォルダを作成 or 確認..."
mkdir -p "$BASE_DIR"

echo ">>> [2/4] uv sync で仮想環境作成＆パッケージ同期..."
echo "    - pyproject.toml / uv.lock を元に"
echo "    - プロジェクト専用環境: $VENV_DIR"
UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync

echo ">>> [3/4] VS Code 設定を書き込み (.vscode/settings.json)..."
mkdir -p "$VSCODE_DIR"

cat > "$SETTINGS_FILE" <<EOF
{
  // この設定ファイルは各マシンで書き換えられる前提ですわ
  // Google Drive で同期しても構いませんが、
  // パスが変わるマシンでは VS Code でインタプリタを選び直してくださいませ。
  "python.defaultInterpreterPath": "$PYTHON_BIN",
  "python.terminal.activateEnvironment": true,
  "jupyter.jupyterServerType": "local",
  "jupyter.notebookFileRoot": "\${workspaceFolder}"
}
EOF

echo ">>> [4/4] 完了しましたわよ♡"
echo ">>> VS Code / Cursor で使う Python はこれですのよ:"
echo "    $PYTHON_BIN"
echo "==============================================="
echo "※ uv.lock はプロジェクト直下に置いたまま Google Drive で共有して頂戴ね。"
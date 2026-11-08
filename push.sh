#!/bin/bash
cd "$(dirname "$0")"
REMOTE="gdrive:colab/$(basename "$PWD")"
rclone sync . "$REMOTE" \
  --exclude ".git/**" --exclude ".venv/**" --exclude "__pycache__/**" \
  --exclude ".ipynb_checkpoints/**" --exclude "outputs/**" --exclude "data/**" \
  --exclude ".env" --exclude ".claude/**" \
  -P "$@"

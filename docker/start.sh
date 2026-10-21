#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export LOCAL_UID="$(id -u)"
export LOCAL_GID="$(id -g)"
compose_files=(-f docker-compose.yml)
case "${1:-}" in
  "") compose_files+=(-f docker-compose.gpu.yml) ;;
  --cpu) ;;
  *) echo "Usage: bash start.sh [--cpu]" >&2; exit 2 ;;
esac

docker compose "${compose_files[@]}" up -d --build
echo 'VS Code: Dev Containers: Attach to Running Container... -> b3cnnpart-notebook-1'
echo 'Open folder: /workspace/B3CNNpart'
echo 'Notebook kernel: Python (B3CNNpart), or /opt/conda/bin/python'

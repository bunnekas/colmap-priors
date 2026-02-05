#!/usr/bin/env bash
set -euo pipefail
module load cuda/12.8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/colmap_priors.env"

SCENE="${1:-}"
if [[ -z "${SCENE}" ]]; then
  echo "Usage: $0 <SCENE_NAME> [--env /path/to/env]"
  exit 1
fi
shift || true

# Allow overriding env path
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_FILE="$2"; shift 2;;
    *) break;;
  esac
done

uv run colmap-priors "${SCENE}" --env "${ENV_FILE}" "$@"

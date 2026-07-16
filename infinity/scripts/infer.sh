#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
nproc="${NPROC_PER_NODE:-1}"
if ((nproc > 1)); then
  exec "${TORCHRUN:-torchrun}" --standalone --nproc_per_node="${nproc}" \
    "${repo_root}/infinity/infer.py" "$@"
fi
exec "${PYTHON:-python}" "${repo_root}/infinity/infer.py" "$@"

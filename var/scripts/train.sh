#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
nproc="${NPROC_PER_NODE:-1}"
translated=()
while (($#)); do
  case "$1" in
    --data-root) translated+=(--data_path "$2"); shift 2 ;;
    --pretrained) translated+=(--pretrain_ckpt "$2"); shift 2 ;;
    --vae-checkpoint) translated+=(--vae_ckpt "$2"); shift 2 ;;
    --output-dir) translated+=(--local_out_dir_path "$2"); shift 2 ;;
    --num-steps) translated+=(--num_gen_steps "$2"); shift 2 ;;
    *) translated+=("$1"); shift ;;
  esac
done

if ((nproc > 1)); then
  exec "${TORCHRUN:-torchrun}" --standalone --nproc_per_node="${nproc}" \
    "${repo_root}/var/train.py" "${translated[@]}"
fi
exec "${PYTHON:-python}" "${repo_root}/var/train.py" "${translated[@]}"

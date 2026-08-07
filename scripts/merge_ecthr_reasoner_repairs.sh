#!/bin/bash
#
# Merge repair JSONL files for train/validation/test back into the main reasoner
# feature files. Run locally or on a login/CPU node after repair jobs finish.
#
# Usage:
#   scripts/merge_ecthr_reasoner_repairs.sh repair_next
#
# Optional:
#   ECTHR_OUT_DIR=outputs/ecthr_b scripts/merge_ecthr_reasoner_repairs.sh repair500_1

set -euo pipefail

ECTHR_OUT_DIR="${ECTHR_OUT_DIR:-outputs/ecthr_b}"
export ECTHR_OUT_DIR
REPAIR_TAG="${1:-repair_next}"

for split in train validation test; do
  base="${ECTHR_OUT_DIR}/reasoner_features_${split}.jsonl"
  repair="${ECTHR_OUT_DIR}/repairs/reasoner_features_${split}_${REPAIR_TAG}.jsonl"

  if [[ ! -f "${repair}" ]]; then
    echo "Skipping ${split}: missing ${repair}" >&2
    continue
  fi

  tmp="${base}.merged_tmp"
  echo "Merging ${repair} into ${base}"
  python -m legal.merge_ecthr_reasoner_features \
    --base "${base}" \
    --repairs "${repair}" \
    --out "${tmp}"
  mv "${tmp}" "${base}"
done

python - <<'PY'
import os
from pathlib import Path
from legal.run_ecthr_reasoner_features import failed_case_ids

for split in ["train", "validation", "test"]:
    path = Path(os.environ["ECTHR_OUT_DIR"]) / f"reasoner_features_{split}.jsonl"
    if path.exists():
        ids = failed_case_ids(path)
        print(f"{split}: {len(ids)} remaining incomplete rows {ids[:50]}")
PY

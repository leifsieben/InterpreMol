#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-/home/ubuntu/interpremol_runs/stage2_full_exactwinner_20260401_185331/checkpoints/best_model.pt}"
OUTPUT_DIR="${2:-/home/ubuntu/interpremol_benchmarks}"
DATA_DIR="${3:-/home/ubuntu/interpremol_benchmark_data}"

cd /home/ubuntu/InterpreMol
source /home/ubuntu/venvs/interpremol/bin/activate

python -m benchmarks.run_moleculenet \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --data-dir "${DATA_DIR}" \
  --datasets bbbp bace clintox hiv muv pcba sider tox21 toxcast esol freesolv lipo qm7 qm8 qm9 \
  --splits random scaffold \
  --seeds 0 1 2 \
  --models interpremol_frozen chemeleon_frozen chemeleon_finetune random_forest \
  --epochs 10 \
  --patience 3 \
  --batch-size 16

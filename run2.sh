#!/bin/bash
#SBATCH --job-name=knit-m-eps-cos
#SBATCH --time=03:00:00
#SBATCH --output=knit-m-eps-cos-%j.out
#SBATCH --error=knit-m-eps-cos-%j.err

set -euo pipefail

# Small-model screening: epsilon prediction + cosine noise schedule.
# Submit as its own single-GPU job:
#   sbatch --gpus=1 run2.sh
export VARIANTS="eps_cos"
export METHOD_UPDATES="${METHOD_UPDATES:-100000}"
export METHOD_SAMPLE_STEPS="${METHOD_SAMPLE_STEPS:-25000 50000 100000}"
export METHOD_BATCH_SIZE="${METHOD_BATCH_SIZE:-32}"
export MAX_SECONDS="${MAX_SECONDS:-10200}"
export METHOD_EXTRA="${METHOD_EXTRA:---grad-clip 1.0 --warmup-updates 5000 --min-lr 1e-5 --lr-schedule cosine --ema-decay 0.9999 --ema-start-step 10000}"
export SEED="${SEED:-42}"

PROJECT="${PROJECT:-/data02/run01/scv7se2/knit_dit}"
exec bash "$PROJECT/run.sh" methods

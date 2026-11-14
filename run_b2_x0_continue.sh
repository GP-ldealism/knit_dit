#!/bin/bash
#SBATCH --job-name=knit-b2-x0-100k
#SBATCH --time=06:00:00
#SBATCH --output=knit-b2-x0-100k-%j.out
#SBATCH --error=knit-b2-x0-100k-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3490054726@qq.com

set -euo pipefail

# Continue the x0_cos B/2 run from 50k to 100k, then sample/evaluate at 100k.
# Submit only after the 50k gate looks acceptable:
#   sbatch --gpus=1 run_b2_x0_continue.sh
# After a 6h timeout: resubmit the same command; it auto-resumes.

MODULE="${MODULE:-miniforge/24.1.2}"
ENV_PATH="${ENV_PATH:-/data02/run01/scv7se2/envs/knit_dit}"
PROJECT="${PROJECT:-/data02/run01/scv7se2/knit_dit}"
OUT="${OUT:-/data02/run01/scv7se2/outputs/knit_dit/11.12}"
RUN_DIR="${RUN_DIR:-$OUT/dit_b2_s42_x0}"
CONFIG="${CONFIG:-$OUT/dit_b2.json}"
FINAL_ROOT="${FINAL_ROOT:-$OUT/b2_100k/x0}"

export MODULE ENV_PATH PROJECT OUT RUN_DIR CONFIG
export UPDATES="${UPDATES:-100000}"
export MAX_SECONDS="${MAX_SECONDS:-19800}"
export TRAIN_EXTRA="${TRAIN_EXTRA:---predict-xstart --noise-schedule squaredcos_cap_v2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export CKPT_EVERY="${CKPT_EVERY:-5000}"
export VAL_EVERY="${VAL_EVERY:-5000}"
export FRESH=0
unset STOP_AT_STEP

module load "$MODULE"
source activate "$ENV_PATH"
cd "$PROJECT"

if [[ ! -f "$RUN_DIR/model.pt" ]]; then
  echo "missing $RUN_DIR/model.pt; run run_b2_x0_gate.sh first" >&2
  exit 1
fi
if [[ ! -f "$OUT/a/requests_s42.npz" ]]; then
  echo "missing $OUT/a/requests_s42.npz; run run0.sh first" >&2
  exit 1
fi

bash "$PROJECT/run.sh" train

LAST_STEP=$(python - "$RUN_DIR/train_summary.json" <<'PY'
import json
import sys
print(int(json.load(open(sys.argv[1], encoding="utf-8")).get("last_step", 0)))
PY
)
echo "training last_step=$LAST_STEP (target=$UPDATES)"

if (( LAST_STEP != UPDATES )); then
  echo "stopped at $LAST_STEP (< $UPDATES); resubmit run_b2_x0_continue.sh to continue" >&2
  exit 0
fi

CKPT="$RUN_DIR/ckpt_$(printf '%06d' "$LAST_STEP").pt"
if [[ ! -f "$CKPT" ]]; then
  CKPT="$RUN_DIR/model.pt"
fi
SAMPLES="${FINAL_SAMPLES:-100}"
DDIM="${FINAL_DDIM:-250}"

python -m knit.sample \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --out "$FINAL_ROOT/sample" \
  --method a0 \
  --num-samples "$SAMPLES" \
  --batch-size 16 \
  --seed 42 \
  --requests "$OUT/a/requests_s42.npz" \
  --palette "$OUT/processed/palette.json" \
  --preview-scale 10 \
  --ddim-steps "$DDIM"

python -m knit.evaluate \
  --run-dir "$FINAL_ROOT/sample" \
  --train-labels "$OUT/processed/train_labels.npy" \
  --out "$FINAL_ROOT/sample/eval.json"

python -m knit.diagnose_denoise \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --labels "$OUT/processed/train_labels.npy" \
  --num-samples "$SAMPLES" \
  --timesteps 100,250,500,750,900 \
  --batch-size 16 \
  --seed 42 \
  --out "$FINAL_ROOT/denoise_step${LAST_STEP}.json"

echo "100k results: $FINAL_ROOT"

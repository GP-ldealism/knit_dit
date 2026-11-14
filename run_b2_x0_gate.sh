#!/bin/bash
#SBATCH --job-name=knit-b2-x0-gate50k
#SBATCH --time=06:00:00
#SBATCH --output=knit-b2-x0-gate50k-%j.out
#SBATCH --error=knit-b2-x0-gate50k-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3490054726@qq.com

set -euo pipefail

# B/2 x0_cos gate: target 100k, stop cleanly at 50k, then sample/evaluate at 50k.
# Submit:              sbatch --gpus=1 run_b2_x0_gate.sh
# After a 6h timeout:  sbatch --gpus=1 run_b2_x0_gate.sh   (auto-resume)
# After 50k passes:    sbatch --gpus=1 run_b2_x0_continue.sh
#
# The gate job stops itself at STOP_AT_STEP; you do not need to monitor or cancel.

MODULE="${MODULE:-miniforge/24.1.2}"
ENV_PATH="${ENV_PATH:-/data02/run01/scv7se2/envs/knit_dit}"
PROJECT="${PROJECT:-/data02/run01/scv7se2/knit_dit}"
OUT="${OUT:-/data02/run01/scv7se2/outputs/knit_dit/11.12}"
RUN_DIR="${RUN_DIR:-$OUT/dit_b2_s42_x0}"
CONFIG="${CONFIG:-$OUT/dit_b2.json}"
STOP_STEP="${STOP_AT_STEP:-50000}"
GATE_ROOT="${GATE_ROOT:-$OUT/b2_gate/x0_step${STOP_STEP}}"

export MODULE ENV_PATH PROJECT OUT RUN_DIR CONFIG
export UPDATES="${UPDATES:-100000}"
export STOP_AT_STEP="$STOP_STEP"
export MAX_SECONDS="${MAX_SECONDS:-19800}"
export TRAIN_EXTRA="${TRAIN_EXTRA:---predict-xstart --noise-schedule squaredcos_cap_v2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export CKPT_EVERY="${CKPT_EVERY:-5000}"
export VAL_EVERY="${VAL_EVERY:-5000}"

if [[ -n "${FRESH:-}" ]]; then
  export FRESH
elif [[ -f "$RUN_DIR/model.pt" ]]; then
  export FRESH=0
elif compgen -G "$RUN_DIR/ckpt_*.pt" > /dev/null; then
  export FRESH=0
  export RESUME="$(ls -1 "$RUN_DIR"/ckpt_*.pt | sort -V | tail -n 1)"
else
  export FRESH=1
fi

module load "$MODULE"
source activate "$ENV_PATH"
cd "$PROJECT"

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
echo "training last_step=$LAST_STEP (target=$UPDATES, gate=$STOP_STEP)"

if (( LAST_STEP != STOP_STEP )); then
  echo "stopped at $LAST_STEP (< $STOP_STEP); resubmit run_b2_x0_gate.sh to continue to $STOP_STEP" >&2
  exit 0
fi

CKPT="$RUN_DIR/ckpt_$(printf '%06d' "$LAST_STEP").pt"
if [[ ! -f "$CKPT" ]]; then
  CKPT="$RUN_DIR/model.pt"
fi
SAMPLES="${GATE_SAMPLES:-100}"
DDIM="${GATE_DDIM:-250}"

python -m knit.sample \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --out "$GATE_ROOT/sample" \
  --method a0 \
  --num-samples "$SAMPLES" \
  --batch-size 16 \
  --seed 42 \
  --requests "$OUT/a/requests_s42.npz" \
  --palette "$OUT/processed/palette.json" \
  --preview-scale 10 \
  --ddim-steps "$DDIM"

python -m knit.evaluate \
  --run-dir "$GATE_ROOT/sample" \
  --train-labels "$OUT/processed/train_labels.npy" \
  --out "$GATE_ROOT/sample/eval.json"

python -m knit.diagnose_denoise \
  --config "$CONFIG" \
  --ckpt "$CKPT" \
  --labels "$OUT/processed/train_labels.npy" \
  --num-samples "$SAMPLES" \
  --timesteps 100,250,500,750,900 \
  --batch-size 16 \
  --seed 42 \
  --out "$GATE_ROOT/denoise_step${LAST_STEP}.json"

echo "50k gate results: $GATE_ROOT"

#!/bin/bash
#SBATCH --job-name=knit-kp-a0
#SBATCH --time=02:00:00
#SBATCH --output=knit-kp-a0-%j.out
#SBATCH --error=knit-kp-a0-%j.err

set -euo pipefail

MODULE="${MODULE:-miniforge/24.1.2}"
ENV_PATH="${ENV_PATH:-/data02/run01/scv7se2/envs/knit_dit}"
PROJECT="${PROJECT:-/data02/run01/scv7se2/knit_dit}"
R="${R:-/data02/run01/scv7se2/outputs/knit_dit/11.12}"
WORK="${WORK:-/data02/run01/scv7se2/outputs/knit_dit/11.23}"
SEED="${SEED:-42}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
DDIM_STEPS="${DDIM_STEPS:-250}"
BATCH_SIZE="${BATCH_SIZE:-16}"
INIT_CKPT="${INIT_CKPT:-$R/dit_b2_s42_x0/ckpt_100000.pt}"
HIST_CONFIG="${HIST_CONFIG:-$R/dit_b2_s42_x0/config.json}"
REQUESTS="${REQUESTS:-$R/a/requests_s42.npz}"
BASE_PROCESSED="${BASE_PROCESSED:-$R/processed}"
KP_PROCESSED="${KP_PROCESSED:-$WORK/processed_kp}"
A_ROOT="${A_ROOT:-$WORK/comparison/a0_s42}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Submit: sbatch --gpus=1 run_kp_step2_a.sh"
  echo "Overrides: MODULE ENV_PATH PROJECT R WORK SEED NUM_SAMPLES DDIM_STEPS BATCH_SIZE"
  echo "           INIT_CKPT HIST_CONFIG REQUESTS BASE_PROCESSED KP_PROCESSED A_ROOT"
  exit 0
fi

module load "$MODULE"
source activate "$ENV_PATH"
cd "$PROJECT"

for path in "$INIT_CKPT" "$HIST_CONFIG" "$REQUESTS" \
  "$BASE_PROCESSED/train_labels.npy" "$BASE_PROCESSED/palette.json"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
done

python -m knit.checkpoint --describe "$INIT_CKPT" \
  --expect-ema --expect-step 100000 \
  --expect-noise-schedule squaredcos_cap_v2 --expect-predict-xstart true

if [[ ! -f "$KP_PROCESSED/train_labels.npy" && -f "${WORK}_kp/processed_kp/train_labels.npy" ]]; then
  KP_PROCESSED="${WORK}_kp/processed_kp"
fi
if [[ ! -f "$KP_PROCESSED/train_labels.npy" ]]; then
  python -m knit.kp --prepare \
    --processed "$BASE_PROCESSED" --kp-out "$KP_PROCESSED"
fi

mkdir -p "$A_ROOT"

python -m knit.sample \
  --config "$HIST_CONFIG" \
  --ckpt "$INIT_CKPT" \
  --out "$A_ROOT/sample" \
  --method a0 \
  --num-samples "$NUM_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --requests "$REQUESTS" \
  --palette "$BASE_PROCESSED/palette.json" \
  --preview-scale 10 \
  --ddim-steps "$DDIM_STEPS" \
  --weights ema

python -m knit.evaluate \
  --run-dir "$A_ROOT/sample" \
  --train-labels "$BASE_PROCESSED/train_labels.npy" \
  --out "$A_ROOT/sample/eval.json"

python -m knit.kp --audit-run \
  --labels "$A_ROOT/sample/labels.npy" \
  --train-all "$BASE_PROCESSED/train_labels.npy" \
  --train-kp "$KP_PROCESSED/train_labels.npy" \
  --out "$A_ROOT/sample/kp_summary.json"

echo "A0 resample complete: $A_ROOT"

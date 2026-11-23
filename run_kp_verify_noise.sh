#!/bin/bash
#SBATCH --job-name=knit-kp-verify
#SBATCH --time=02:00:00
#SBATCH --output=knit-kp-verify-%j.out
#SBATCH --error=knit-kp-verify-%j.err

set -euo pipefail

MODULE="${MODULE:-miniforge/24.1.2}"
ENV_PATH="${ENV_PATH:-/data02/run01/scv7se2/envs/knit_dit}"
PROJECT="${PROJECT:-/data02/run01/scv7se2/knit_dit}"
R="${R:-/data02/run01/scv7se2/outputs/knit_dit/11.12}"
WORK="${WORK:-/data02/run01/scv7se2/outputs/knit_dit/11.23}"
SEEDS="${SEEDS:-43 44}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
DDIM_STEPS="${DDIM_STEPS:-250}"
BATCH_SIZE="${BATCH_SIZE:-16}"
INIT_CKPT="${INIT_CKPT:-$R/dit_b2_s42_x0/ckpt_100000.pt}"
HIST_CONFIG="${HIST_CONFIG:-$R/dit_b2_s42_x0/config.json}"
BASE_PROCESSED="${BASE_PROCESSED:-$R/processed}"
KP_PROCESSED="${KP_PROCESSED:-$WORK/processed_kp}"
B_CONFIG="${B_CONFIG:-$WORK/ft_full/config.json}"
B_CKPT="${B_CKPT:-$WORK/ft_full/train/ckpt_000500.pt}"
C_CONFIG="${C_CONFIG:-$WORK/ft_kp/config.json}"
C_CKPT="${C_CKPT:-$WORK/ft_kp/train/ckpt_000500.pt}"
VROOT="${VROOT:-$WORK/verify_noise}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Submit: sbatch --gpus=1 run_kp_verify_noise.sh"
  echo "Resample fixed A/B500/C500 weights on new noise batches (default seeds 43 44)."
  echo "Overrides: MODULE ENV_PATH PROJECT R WORK SEEDS NUM_SAMPLES DDIM_STEPS BATCH_SIZE"
  echo "           INIT_CKPT HIST_CONFIG BASE_PROCESSED KP_PROCESSED B_CONFIG B_CKPT C_CONFIG C_CKPT VROOT"
  exit 0
fi

module load "$MODULE"
source activate "$ENV_PATH"
cd "$PROJECT"

if [[ ! -f "$KP_PROCESSED/train_labels.npy" && -f "${WORK}_kp/processed_kp/train_labels.npy" ]]; then
  KP_PROCESSED="${WORK}_kp/processed_kp"
fi

for path in "$INIT_CKPT" "$HIST_CONFIG" "$B_CONFIG" "$B_CKPT" "$C_CONFIG" "$C_CKPT" \
  "$BASE_PROCESSED/train_labels.npy" "$BASE_PROCESSED/palette.json" "$KP_PROCESSED/train_labels.npy"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
done

for seed in $SEEDS; do
  req="$VROOT/requests_s${seed}.npz"
  if [[ ! -f "$req" ]]; then
    python -m knit.requests --config "$HIST_CONFIG" \
      --num-samples "$NUM_SAMPLES" --seed "$seed" --out "$req"
  fi
  for group in a b500 c500; do
    case "$group" in
      a) cfg="$HIST_CONFIG"; ckpt="$INIT_CKPT" ;;
      b500) cfg="$B_CONFIG"; ckpt="$B_CKPT" ;;
      c500) cfg="$C_CONFIG"; ckpt="$C_CKPT" ;;
    esac
    out="$VROOT/${group}_s${seed}/sample"
    mkdir -p "$out"
    python -m knit.sample \
      --config "$cfg" \
      --ckpt "$ckpt" \
      --out "$out" \
      --method a0 \
      --num-samples "$NUM_SAMPLES" \
      --batch-size "$BATCH_SIZE" \
      --seed "$seed" \
      --requests "$req" \
      --palette "$BASE_PROCESSED/palette.json" \
      --preview-scale 10 \
      --ddim-steps "$DDIM_STEPS" \
      --weights ema
    python -m knit.evaluate \
      --run-dir "$out" \
      --train-labels "$BASE_PROCESSED/train_labels.npy" \
      --out "$out/eval.json"
    python -m knit.kp --audit-run \
      --labels "$out/labels.npy" \
      --train-all "$BASE_PROCESSED/train_labels.npy" \
      --train-kp "$KP_PROCESSED/train_labels.npy" \
      --out "$out/kp_summary.json"
    echo "done $group seed=$seed -> $out"
  done
done

echo "verify_noise complete: $VROOT (seeds: $SEEDS)"

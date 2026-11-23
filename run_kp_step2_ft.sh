#!/bin/bash
#SBATCH --job-name=knit-kp-ft
#SBATCH --time=03:00:00
#SBATCH --output=knit-kp-ft-%j.out
#SBATCH --error=knit-kp-ft-%j.err

set -euo pipefail

MODULE="${MODULE:-miniforge/24.1.2}"
ENV_PATH="${ENV_PATH:-/data02/run01/scv7se2/envs/knit_dit}"
PROJECT="${PROJECT:-/data02/run01/scv7se2/knit_dit}"
R="${R:-/data02/run01/scv7se2/outputs/knit_dit/11.12}"
WORK="${WORK:-/data02/run01/scv7se2/outputs/knit_dit/11.23}"
GROUP="${1:-${GROUP:-b}}"
SEED="${SEED:-42}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
DDIM_STEPS="${DDIM_STEPS:-250}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
UPDATES="${UPDATES:-1000}"
LR="${LR:-1e-5}"
EMA_DECAY="${EMA_DECAY:-0.99}"
SAMPLE_STEPS="${SAMPLE_STEPS:-500 1000}"
INIT_CKPT="${INIT_CKPT:-$R/dit_b2_s42_x0/ckpt_100000.pt}"
HIST_CONFIG="${HIST_CONFIG:-$R/dit_b2_s42_x0/config.json}"
REQUESTS="${REQUESTS:-$R/a/requests_s42.npz}"
BASE_PROCESSED="${BASE_PROCESSED:-$R/processed}"
KP_PROCESSED="${KP_PROCESSED:-$WORK/processed_kp}"

if [[ "$GROUP" == "-h" || "$GROUP" == "--help" ]]; then
  echo "Submit: sbatch --gpus=1 run_kp_step2_ft.sh b"
  echo "        sbatch --gpus=1 run_kp_step2_ft.sh c"
  echo "b = full-data fine-tune -> \$WORK/ft_full"
  echo "c = K/P-data fine-tune  -> \$WORK/ft_kp"
  echo "Overrides: MODULE ENV_PATH PROJECT R WORK SEED NUM_SAMPLES DDIM_STEPS BATCH_SIZE"
  echo "           TRAIN_BATCH_SIZE NUM_WORKERS UPDATES LR EMA_DECAY SAMPLE_STEPS"
  echo "           INIT_CKPT HIST_CONFIG REQUESTS BASE_PROCESSED KP_PROCESSED"
  exit 0
fi

case "$GROUP" in
  b)
    DATA_DIR="$BASE_PROCESSED"
    FT_ROOT="$WORK/ft_full"
    ;;
  c)
    DATA_DIR="$KP_PROCESSED"
    FT_ROOT="$WORK/ft_kp"
    ;;
  *)
    echo "unknown group: $GROUP (expected b or c)" >&2
    exit 2
    ;;
esac

CONFIG="$FT_ROOT/config.json"
TRAIN_DIR="$FT_ROOT/train"

module load "$MODULE"
source activate "$ENV_PATH"
cd "$PROJECT"

for path in "$INIT_CKPT" "$HIST_CONFIG" "$REQUESTS" \
  "$BASE_PROCESSED/train_labels.npy" "$BASE_PROCESSED/test_labels.npy" \
  "$BASE_PROCESSED/palette.json"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
done

if [[ ! -f "$KP_PROCESSED/train_labels.npy" && -f "${WORK}_kp/processed_kp/train_labels.npy" ]]; then
  KP_PROCESSED="${WORK}_kp/processed_kp"
fi
if [[ ! -f "$KP_PROCESSED/train_labels.npy" ]]; then
  python -m knit.kp --prepare \
    --processed "$BASE_PROCESSED" --kp-out "$KP_PROCESSED"
fi
if [[ "$GROUP" == "c" ]]; then
  DATA_DIR="$KP_PROCESSED"
fi

for path in "$DATA_DIR/train_labels.npy" "$DATA_DIR/train_meta.json" \
  "$DATA_DIR/val_labels.npy" "$DATA_DIR/val_meta.json"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required data file: $path" >&2
    exit 1
  fi
done

echo "group=$GROUP data_dir=$DATA_DIR ft_root=$FT_ROOT"

python -m knit.checkpoint --describe "$INIT_CKPT" \
  --expect-ema --expect-step 100000 \
  --expect-noise-schedule squaredcos_cap_v2 --expect-predict-xstart true

python -m knit.finetune_config \
  --source "$HIST_CONFIG" \
  --out "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --base-data-dir "$BASE_PROCESSED" \
  --updates "$UPDATES" \
  --lr "$LR" \
  --warmup-updates 0 \
  --lr-schedule constant \
  --ema-decay "$EMA_DECAY" \
  --ema-start-step 0 \
  --val-every 500 \
  --ckpt-every 500

mkdir -p "$TRAIN_DIR"

python -m knit.train \
  --config "$CONFIG" \
  --out "$TRAIN_DIR" \
  --seed "$SEED" \
  --updates "$UPDATES" \
  --batch-size "$TRAIN_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --init-from "$INIT_CKPT"

for step in $SAMPLE_STEPS; do
  if (( step > UPDATES )); then
    continue
  fi
  ckpt="$TRAIN_DIR/ckpt_$(printf '%06d' "$step").pt"
  if [[ ! -f "$ckpt" ]]; then
    ckpt="$TRAIN_DIR/model.pt"
  fi
  sample_dir="$FT_ROOT/sample_step${step}"
  python -m knit.sample \
    --config "$CONFIG" \
    --ckpt "$ckpt" \
    --out "$sample_dir" \
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
    --run-dir "$sample_dir" \
    --train-labels "$BASE_PROCESSED/train_labels.npy" \
    --out "$sample_dir/eval.json"
  python -m knit.kp --audit-run \
    --labels "$sample_dir/labels.npy" \
    --train-all "$BASE_PROCESSED/train_labels.npy" \
    --train-kp "$KP_PROCESSED/train_labels.npy" \
    --out "$sample_dir/kp_summary.json"
done

echo "group $GROUP fine-tune complete: $FT_ROOT"

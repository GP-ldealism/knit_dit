#!/bin/bash
#SBATCH --job-name=knit-thin-v
#SBATCH --time=04:00:00
#SBATCH --output=knit-thin-v-%j.out
#SBATCH --error=knit-thin-v-%j.err

set -euo pipefail

MODULE="${MODULE:-miniforge/24.1.2}"
ENV_PATH="${ENV_PATH:-/data02/run01/scv7se2/envs/knit_dit}"
PROJECT="${PROJECT:-/data02/run01/scv7se2/knit_dit}"
R="${R:-/data02/run01/scv7se2/outputs/knit_dit/11.12}"
WORK="${WORK:-/data02/run01/scv7se2/outputs/knit_dit/11.23}"
PHASE="${PHASE:-pilot}"
SEEDS="${SEEDS:-}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
DDIM_STEPS="${DDIM_STEPS:-250}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CONFIG="${CONFIG:-$WORK/ft_kp/config.json}"
CKPT="${CKPT:-$WORK/ft_kp/train/ckpt_000500.pt}"
PROCESSED="${PROCESSED:-$R/processed}"
KP_PROCESSED="${KP_PROCESSED:-$WORK/processed_kp}"
GUIDE_ROOT="${GUIDE_ROOT:-$WORK/thin_v_guide}"
GUIDE_STEPS="${GUIDE_STEPS:-80,100,120,140,160,180,200,220}"
GUIDE_TEMPERATURE="${GUIDE_TEMPERATURE:-4.0}"
RULE_THRESHOLD="${RULE_THRESHOLD:-0.7}"
PILOT_STRENGTHS="${PILOT_STRENGTHS:-0.1 0.5}"
FORMAL_STRENGTH="${FORMAL_STRENGTH:-}"
PALETTE="$PROCESSED/palette.json"
TRAIN_LABELS="$PROCESSED/train_labels.npy"
TRAIN_KP="$KP_PROCESSED/train_labels.npy"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Submit pilot: sbatch --gpus=1 run_thin_v_guide.sh"
  echo "Submit formal: FORMAL_STRENGTH=<value> PHASE=formal sbatch --gpus=1 run_thin_v_guide.sh"
  echo "Defaults: pilot=seeds 45 46, 20 samples; formal=seeds 47 48, 100 samples"
  echo "Overrides: MODULE ENV_PATH PROJECT R WORK PHASE SEEDS NUM_SAMPLES DDIM_STEPS BATCH_SIZE"
  echo "           CONFIG CKPT PROCESSED KP_PROCESSED GUIDE_ROOT GUIDE_STEPS GUIDE_TEMPERATURE"
  echo "           RULE_THRESHOLD PILOT_STRENGTHS FORMAL_STRENGTH"
  exit 0
fi

case "$PHASE" in
  pilot|formal) ;;
  *) echo "PHASE must be pilot or formal" >&2; exit 1 ;;
esac
if [[ "$PHASE" == "formal" && -z "$FORMAL_STRENGTH" ]]; then
  echo "FORMAL_STRENGTH is required when PHASE=formal" >&2
  exit 1
fi
if [[ -z "$SEEDS" ]]; then
  if [[ "$PHASE" == "pilot" ]]; then
    SEEDS="45 46"
  else
    SEEDS="47 48"
  fi
fi
if [[ -z "$NUM_SAMPLES" ]]; then
  if [[ "$PHASE" == "pilot" ]]; then
    NUM_SAMPLES=20
  else
    NUM_SAMPLES=100
  fi
fi

module load "$MODULE"
source activate "$ENV_PATH"
cd "$PROJECT"

if [[ ! -f "$TRAIN_KP" && -f "${WORK}_kp/processed_kp/train_labels.npy" ]]; then
  KP_PROCESSED="${WORK}_kp/processed_kp"
  TRAIN_KP="$KP_PROCESSED/train_labels.npy"
fi

for path in "$CONFIG" "$CKPT" "$PALETTE" "$TRAIN_LABELS" "$TRAIN_KP"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
done

make_request() {
  local req="$1"
  local seed="$2"
  if [[ ! -f "$req" ]]; then
    python -m knit.requests --config "$CONFIG" \
      --num-samples "$NUM_SAMPLES" --seed "$seed" --out "$req"
  fi
}

run_sample() {
  local out="$1"
  local req="$2"
  local guide_style="$3"
  local strength="$4"
  mkdir -p "$out"
  python -m knit.sample \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --out "$out" \
    --method a0 \
    --num-samples "$NUM_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --seed 0 \
    --requests "$req" \
    --palette "$PALETTE" \
    --preview-scale 10 \
    --ddim-steps "$DDIM_STEPS" \
    --weights ema \
    --guide-style "$guide_style" \
    --guide-strength "$strength" \
    --guide-steps "$GUIDE_STEPS" \
    --guide-temperature "$GUIDE_TEMPERATURE" \
    --rule-threshold "$RULE_THRESHOLD"
  python -m knit.evaluate \
    --run-dir "$out" \
    --train-labels "$TRAIN_LABELS" \
    --out "$out/eval.json"
  python -m knit.kp --audit-run \
    --labels "$out/labels.npy" \
    --train-all "$TRAIN_LABELS" \
    --train-kp "$TRAIN_KP" \
    --out "$out/kp_summary.json"
}

if [[ "$PHASE" == "pilot" ]]; then
  strengths="$PILOT_STRENGTHS"
else
  strengths="$FORMAL_STRENGTH"
fi

for seed in $SEEDS; do
  seed_dir="$GUIDE_ROOT/$PHASE/seed${seed}"
  request_dir="$GUIDE_ROOT/requests"
  mkdir -p "$seed_dir" "$request_dir"

  req0="$request_dir/requests_s${seed}_c0.npz"
  make_request "$req0" "$seed"

  a_out="$seed_dir/a"
  run_sample "$a_out" "$req0" none 0

  b_inputs=()
  for candidate in 0 1 2 3; do
    req="$request_dir/requests_s${seed}_c${candidate}.npz"
    candidate_seed=$((seed + 100 * candidate))
    make_request "$req" "$candidate_seed"
    b_candidate_out="$seed_dir/b_c${candidate}"
    run_sample "$b_candidate_out" "$req" none 0
    b_inputs+=("$b_candidate_out/labels.npy")
  done
  b_out="$seed_dir/b"
  python -m knit.guidance select \
    --inputs "${b_inputs[@]}" \
    --out "$b_out" \
    --palette "$PALETTE"
  python -m knit.evaluate \
    --run-dir "$b_out" \
    --train-labels "$TRAIN_LABELS" \
    --out "$b_out/eval.json"
  python -m knit.kp --audit-run \
    --labels "$b_out/labels.npy" \
    --train-all "$TRAIN_LABELS" \
    --train-kp "$TRAIN_KP" \
    --out "$b_out/kp_summary.json"

  c_runs=()
  for strength in $strengths; do
    name="c_${strength//./p}"
    c_out="$seed_dir/$name"
    run_sample "$c_out" "$req0" thin_v "$strength"
    c_runs+=("$name=$c_out")
  done

  compare_args=(--run "a=$a_out" --run "b=$b_out")
  for item in "${c_runs[@]}"; do
    compare_args+=(--run "$item")
  done
  python -m knit.guidance compare \
    "${compare_args[@]}" \
    --threshold "$RULE_THRESHOLD" \
    --out "$seed_dir/compare.json"
  echo "done phase=$PHASE seed=$seed -> $seed_dir"
done

echo "thin_v_guide complete: $GUIDE_ROOT/$PHASE"

#!/bin/bash
#SBATCH --job-name=knit-a
#SBATCH --time=06:00:00
#SBATCH --output=knit-a-%j.out
#SBATCH --error=knit-a-%j.err

set -euo pipefail

MODULE="${MODULE:-miniforge/24.1.2}"
ENV_PATH="${ENV_PATH:-/data02/run01/scv7se2/envs/knit_dit}"
PROJECT="${PROJECT:-/data02/run01/scv7se2/knit_dit}"
DATA="${DATA:-/data02/run01/scv7se2/datasets/knit}"
OUT="${OUT:-/data02/run01/scv7se2/outputs/knit_dit/11.12}"

SEED="${SEED:-42}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
DIAG_SAMPLES="${DIAG_SAMPLES:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
UPDATES="${UPDATES:-100000}"
MAX_SECONDS="${MAX_SECONDS:-19800}"
DDIM_STEPS="${DDIM_STEPS:-250}"
WEIGHTS="${WEIGHTS:-auto}"
STEPS="${STEPS:-100 250 500}"
METHODS="${METHODS:-a0}"
VARIANTS="${VARIANTS:-eps_linear eps_cos x0_cos v_cos eps_cos_logitnorm}"
METHOD_UPDATES="${METHOD_UPDATES:-100000}"
METHOD_BATCH_SIZE="${METHOD_BATCH_SIZE:-32}"
METHOD_SAMPLE_STEPS="${METHOD_SAMPLE_STEPS:-}"
TRAIN_EXTRA="${TRAIN_EXTRA:-}"
METHOD_EXTRA="${METHOD_EXTRA:-}"
STAGE="${1:-${STAGE:-help}}"

module load "$MODULE"
source activate "$ENV_PATH"

cd "$PROJECT"
mkdir -p "$OUT/processed" "$OUT/splits" "$OUT/a" "$OUT/diag" "$OUT/methods"

RAW="${RAW:-$DATA/instruction}"

usage() {
  cat <<'EOF'
Usage:
  sbatch --gpus=1 run.sh prep
  sbatch --gpus=1 run.sh diag
  sbatch --gpus=1 run.sh methods
  sbatch --gpus=1 run.sh speed
  sbatch --gpus=1 run.sh train
  sbatch --gpus=1 run.sh sample
  sbatch --gpus=1 run.sh eval

Stages:
  prep    audit/split/encode NIK instruction data and write server configs
  diag    DDIM 100/250/500 on the existing 100k checkpoint + denoise diagnosis
  methods small-DiT method sweep (schedule/target/t-sampling); train+sample+diagnose per variant
  speed   B/2 1000-step throughput test
  train   B/2 100000-step training; resubmit to resume from model.pt
  sample  B/2 A0 sampling with the selected checkpoint and DDIM steps
  eval    evaluate sampled A0 runs

Common overrides:
  SEED=42 NUM_SAMPLES=100 DIAG_SAMPLES=100 REQUEST_SAMPLES=1000 BATCH_SIZE=16
  UPDATES=100000 MAX_SECONDS=19800 DDIM_STEPS=250 WEIGHTS=auto
  STEPS="100 250 500" METHODS=a0
  VARIANTS="eps_linear eps_cos x0_cos v_cos eps_cos_logitnorm" METHOD_UPDATES=100000
  METHOD_SAMPLE_STEPS="25000 50000 100000"
  METHOD_EXTRA="--grad-clip 1.0 --warmup-updates 5000 --min-lr 1e-5 --lr-schedule cosine --ema-decay 0.9999 --ema-start-step 10000"
  TRAIN_EXTRA="--noise-schedule squaredcos_cap_v2"
  PROJECT=/path/to/knit_dit OUT=/path/to/output ENV_PATH=/path/to/env
  DATA=/path/to/datasets/knit RAW=/path/to/instruction
  CKPT=/path/to/model.pt CONFIG=/path/to/config.json
  RESUME=/path/to/model.pt RUN_DIR=/path/to/train_run FRESH=1
  SAMPLE_ROOT=/path/to/sample_runs
  SPEED_UPDATES=1000 SPEED_MAX_SECONDS=3600 SPEED_BATCH_SIZE=32
  TRAIN_BATCH_SIZE=32 NUM_WORKERS=4 GRAD_ACCUM_STEPS=1 AMP=none|fp16
  VAL_EVERY=5000 CKPT_EVERY=5000
EOF
}

write_config() {
  local src="$1"
  local dst="$2"
  local tmp="$dst.tmp.$$"
  python - "$src" "$tmp" "$OUT/processed" <<'PY'
import json
import sys

src, dst, processed = sys.argv[1:]
with open(src, encoding="utf-8") as f:
    config = json.load(f)

config["data_dir"] = processed
config["data"] = {
    "train_labels": f"{processed}/train_labels.npy",
    "train_meta": f"{processed}/train_meta.json",
    "val_labels": f"{processed}/val_labels.npy",
    "val_meta": f"{processed}/val_meta.json",
    "test_labels": f"{processed}/test_labels.npy",
    "test_meta": f"{processed}/test_meta.json",
    "palette": f"{processed}/palette.json",
}

with open(dst, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
PY
  mv "$tmp" "$dst"
}

ensure_processed() {
  if [[ -f "$OUT/processed/train_labels.npy" &&
        -f "$OUT/processed/train_meta.json" &&
        -f "$OUT/processed/val_labels.npy" &&
        -f "$OUT/processed/val_meta.json" &&
        -f "$OUT/processed/test_labels.npy" &&
        -f "$OUT/processed/test_meta.json" &&
        -f "$OUT/processed/palette.json" ]]; then
    return
  fi
  if [[ ! -d "$RAW" ]]; then
    echo "missing NIK instruction directory: $RAW" >&2
    exit 1
  fi
  mkdir -p "$OUT/splits" "$OUT/processed"
  if [[ ! -f "$OUT/audit.json" ]]; then
    python -m knit.data --audit --root "$RAW" --out "$OUT/audit.json"
  fi
  if [[ ! -f "$OUT/splits/splits.json" ]]; then
    python -m knit.data --split --root "$RAW" --out "$OUT/splits/splits.json"
  fi
  python -m knit.data --encode --root "$RAW" \
    --splits "$OUT/splits/splits.json" --out "$OUT/processed"
}

ensure_configs() {
  write_config "$PROJECT/configs/base.json" "$OUT/base.json"
  write_config "$PROJECT/configs/dit_b2.json" "$OUT/dit_b2.json"
}

ensure_requests() {
  local count="${REQUEST_SAMPLES:-$NUM_SAMPLES}"
  if (( NUM_SAMPLES > count )); then
    count="$NUM_SAMPLES"
  fi
  if (( DIAG_SAMPLES > count )); then
    count="$DIAG_SAMPLES"
  fi
  local path="$OUT/a/requests_s${SEED}.npz"
  if [[ -f "$path" ]]; then
    local have
    if ! have=$(python - "$path" <<'PY'
import sys
import numpy as np
payload = np.load(sys.argv[1], allow_pickle=False)
print(payload["noise"].shape[0] if "noise" in payload else payload["xT"].shape[0])
PY
    ); then
      have=0
    fi
    if (( have >= count )); then
      return
    fi
  fi
  local tmp="$OUT/a/requests_s${SEED}.tmp.$$.npz"
  python -m knit.requests \
    --config "$OUT/base.json" \
    --num-samples "$count" \
    --seed "$SEED" \
    --out "$tmp"
  mv "$tmp" "$path"
  mv "${tmp%.npz}.json" "$OUT/a/requests_s${SEED}.json"
}

run_prep() {
  ensure_processed
  ensure_configs
  ensure_requests
}

run_diag() {
  ensure_processed
  ensure_configs
  ensure_requests
  local ckpt="${CKPT:-$OUT/base_s${SEED}/model.pt}"
  if [[ ! -f "$ckpt" ]]; then
    echo "missing 100k checkpoint: $ckpt" >&2
    exit 1
  fi
  for steps in $STEPS; do
    local run_dir="$OUT/diag/a0_ddim${steps}"
    python -m knit.sample \
      --config "$OUT/base.json" \
      --ckpt "$ckpt" \
      --out "$run_dir" \
      --method a0 \
      --num-samples "$DIAG_SAMPLES" \
      --batch-size "$BATCH_SIZE" \
      --seed "$SEED" \
      --requests "$OUT/a/requests_s${SEED}.npz" \
      --palette "$OUT/processed/palette.json" \
      --preview-scale 10 \
      --ddim-steps "$steps"
    python -m knit.evaluate \
      --run-dir "$run_dir" \
      --train-labels "$OUT/processed/train_labels.npy" \
      --out "$run_dir/eval.json"
  done
  python -m knit.diagnose_denoise \
    --config "$OUT/base.json" \
    --ckpt "$ckpt" \
    --labels "$OUT/processed/train_labels.npy" \
    --num-samples "$DIAG_SAMPLES" \
    --timesteps 100,250,500,750,900 \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --out "$OUT/diag/denoise_s${SEED}.json"
}

run_speed() {
  ensure_processed
  ensure_configs
  local run_dir="$OUT/diag/b2_speed"
  local resume="${RESUME:-}"
  if [[ "${FRESH:-0}" == "1" ]]; then
    resume=""
  elif [[ -z "$resume" && -f "$run_dir/model.pt" ]]; then
    resume="$run_dir/model.pt"
  fi
  local args=(
    --config "$OUT/dit_b2.json"
    --out "$run_dir"
    --seed "$SEED"
    --updates "${SPEED_UPDATES:-1000}"
    --batch-size "${SPEED_BATCH_SIZE:-32}"
    --num-workers "${NUM_WORKERS:-4}"
    --grad-accum-steps "${GRAD_ACCUM_STEPS:-1}"
    --val-every 500
    --ckpt-every 500
    --max-seconds "${SPEED_MAX_SECONDS:-3600}"
  )
  if [[ -n "${AMP:-}" ]]; then
    args+=(--amp "$AMP")
  fi
  if [[ -n "$resume" ]]; then
    args+=(--resume "$resume")
  fi
  python -m knit.train "${args[@]}"
}

run_train() {
  ensure_processed
  ensure_configs
  local run_dir="${RUN_DIR:-$OUT/dit_b2_s${SEED}}"
  local resume="${RESUME:-}"
  if [[ "${FRESH:-0}" == "1" ]]; then
    resume=""
  elif [[ -z "$resume" && -f "$run_dir/model.pt" ]]; then
    resume="$run_dir/model.pt"
  fi
  local args=(
    --config "$OUT/dit_b2.json"
    --out "$run_dir"
    --seed "$SEED"
    --updates "$UPDATES"
    --batch-size "${TRAIN_BATCH_SIZE:-32}"
    --num-workers "${NUM_WORKERS:-4}"
    --grad-accum-steps "${GRAD_ACCUM_STEPS:-1}"
    --val-every "${VAL_EVERY:-5000}"
    --ckpt-every "${CKPT_EVERY:-5000}"
    --max-seconds "$MAX_SECONDS"
  )
  if [[ -n "${AMP:-}" ]]; then
    args+=(--amp "$AMP")
  fi
  if [[ -n "$TRAIN_EXTRA" ]]; then
    args+=($TRAIN_EXTRA)
  fi
  if [[ -n "$resume" ]]; then
    args+=(--resume "$resume")
  fi
  python -m knit.train "${args[@]}"
}

variant_flags() {
  case "$1" in
    eps_linear) echo "" ;;
    eps_cos) echo "--noise-schedule squaredcos_cap_v2" ;;
    eps_cos_logitnorm) echo "--noise-schedule squaredcos_cap_v2 --t-sample logitnorm" ;;
    eps_cos_minsnr) echo "--noise-schedule squaredcos_cap_v2 --min-snr-gamma 5" ;;
    x0_linear) echo "--predict-xstart" ;;
    x0_cos) echo "--predict-xstart --noise-schedule squaredcos_cap_v2" ;;
    v_linear) echo "--predict-v" ;;
    v_cos) echo "--predict-v --noise-schedule squaredcos_cap_v2" ;;
    *)
      echo "unknown variant: $1" >&2
      exit 2
      ;;
  esac
}

run_methods() {
  if [[ ! -f "$OUT/processed/train_labels.npy" ]]; then
    echo "processed data missing: run 'sbatch --gpus=1 run0.sh' first" >&2
    exit 1
  fi
  ensure_configs
  ensure_requests
  local config="$OUT/base.json"
  local sample_steps="${METHOD_SAMPLE_STEPS:-$METHOD_UPDATES}"
  for variant in $VARIANTS; do
    local run_dir="$OUT/methods/${variant}"
    local flags
    flags="$(variant_flags "$variant")"
    local train_args=(
      --config "$config"
      --out "$run_dir"
      --seed "$SEED"
      --updates "$METHOD_UPDATES"
      --batch-size "$METHOD_BATCH_SIZE"
      --num-workers "${NUM_WORKERS:-4}"
      --grad-accum-steps "${GRAD_ACCUM_STEPS:-1}"
      --val-every 5000
      --ckpt-every 5000
      --max-seconds "$MAX_SECONDS"
    )
    if [[ -n "${AMP:-}" ]]; then
      train_args+=(--amp "$AMP")
    fi
    if [[ -n "$flags" ]]; then
      train_args+=($flags)
    fi
    if [[ -n "$METHOD_EXTRA" ]]; then
      train_args+=($METHOD_EXTRA)
    fi
    python -m knit.train "${train_args[@]}"

    local step
    for step in $sample_steps; do
      if (( step > METHOD_UPDATES )); then
        continue
      fi
      local ckpt_file="$run_dir/model.pt"
      if (( step < METHOD_UPDATES )); then
        ckpt_file="$run_dir/ckpt_$(printf '%06d' "$step").pt"
      fi
      if [[ ! -f "$ckpt_file" ]]; then
        echo "skip ${variant} step ${step}: missing $ckpt_file" >&2
        continue
      fi
      local sample_dir="$OUT/methods/${variant}/sample_step${step}_ddim${DDIM_STEPS}"
      python -m knit.sample \
        --config "$config" \
        --ckpt "$ckpt_file" \
        --out "$sample_dir" \
        --method a0 \
        --num-samples "$DIAG_SAMPLES" \
        --batch-size "$BATCH_SIZE" \
        --seed "$SEED" \
        --requests "$OUT/a/requests_s${SEED}.npz" \
        --palette "$OUT/processed/palette.json" \
        --preview-scale 10 \
        --ddim-steps "$DDIM_STEPS"
      python -m knit.evaluate \
        --run-dir "$sample_dir" \
        --train-labels "$OUT/processed/train_labels.npy" \
        --out "$sample_dir/eval.json"
      python -m knit.diagnose_denoise \
        --config "$config" \
        --ckpt "$ckpt_file" \
        --labels "$OUT/processed/train_labels.npy" \
        --num-samples "$DIAG_SAMPLES" \
        --timesteps 100,250,500,750,900 \
        --batch-size "$BATCH_SIZE" \
        --seed "$SEED" \
        --out "$run_dir/denoise_step${step}_s${SEED}.json"
    done
  done
}

run_sample() {
  ensure_processed
  ensure_configs
  ensure_requests
  local ckpt="${CKPT:-$OUT/dit_b2_s${SEED}/model.pt}"
  local config="${CONFIG:-$OUT/dit_b2.json}"
  local root="${SAMPLE_ROOT:-$OUT/b2/a}"
  if [[ ! -f "$ckpt" ]]; then
    echo "missing checkpoint: $ckpt" >&2
    exit 1
  fi
  for method in $METHODS; do
    local run_dir="$root/${method}_s${SEED}_ddim${DDIM_STEPS}"
    local args=(
      --config "$config"
      --ckpt "$ckpt"
      --out "$run_dir"
      --method "$method"
      --num-samples "$NUM_SAMPLES"
      --batch-size "$BATCH_SIZE"
      --seed "$SEED"
      --requests "$OUT/a/requests_s${SEED}.npz"
      --palette "$OUT/processed/palette.json"
      --preview-scale 10
      --ddim-steps "$DDIM_STEPS"
      --weights "$WEIGHTS"
    )
    if [[ "$method" != "a0" ]]; then
      args+=(--candidate-budget 24)
    fi
    if [[ "$method" == "a2" ]]; then
      args+=(--feedback-steps 70,85,95)
    fi
    python -m knit.sample "${args[@]}"
  done
}

run_eval() {
  ensure_processed
  local root="${SAMPLE_ROOT:-$OUT/b2/a}"
  for method in $METHODS; do
    local run_dir="$root/${method}_s${SEED}_ddim${DDIM_STEPS}"
    python -m knit.evaluate \
      --run-dir "$run_dir" \
      --train-labels "$OUT/processed/train_labels.npy" \
      --out "$run_dir/eval.json"
  done
}

case "$STAGE" in
  prep) run_prep ;;
  diag) run_diag ;;
  speed) run_speed ;;
  train) run_train ;;
  methods) run_methods ;;
  sample) run_sample ;;
  eval) run_eval ;;
  help|-h|--help) usage ;;
  *)
    echo "unknown stage: $STAGE" >&2
    usage
    exit 2
    ;;
esac

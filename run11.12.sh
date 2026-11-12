#!/bin/bash
#SBATCH --job-name=knit-11.12
#SBATCH --time=06:00:00
#SBATCH --output=run11.12-%j.out
#SBATCH --error=run11.12-%j.err

# 用法：
#   sbatch --gpus=1 run11.12.sh


set -euo pipefail

module load miniforge/24.1.2
source activate /data02/run01/scv7se2/envs/knit_dit

# date
# tar -xf /data02/run01/scv7se2/datasets/knit.tar.gz -C /dev/shm
# date

PROJECT=/data02/run01/scv7se2/knit_dit
DATA=/data02/run01/scv7se2/datasets/knit
# DATA=/dev/shm/knit
OUT=/data02/run01/scv7se2/outputs/knit_dit/11.12

SEED=42
NUM_SAMPLES=100
BATCH_SIZE=16

cd "$PROJECT"

mkdir -p "$OUT/a" "$OUT/processed" "$OUT/splits"

# NIK 原始 instruction 图；A 组只允许该目录，禁止回退到 $DATA。
RAW="$DATA/instruction"
if [[ ! -d "$RAW" ]]; then
  echo "missing NIK instruction directory: $RAW" >&2
  exit 1
fi

python - "$PROJECT/configs/base.json" "$OUT/base.json" "$OUT/processed" <<'PY'
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

python -m knit.data --audit --root "$RAW" --out "$OUT/audit.json"
python -m knit.data --split --root "$RAW" --out "$OUT/splits/splits.json"
python -m knit.data --encode --root "$RAW" \
  --splits "$OUT/splits/splits.json" --out "$OUT/processed"

python -m knit.train \
  --config "$OUT/base.json" \
  --seed "$SEED" \
  --out "$OUT/base_s${SEED}"

python -m knit.requests \
  --config "$OUT/base.json" \
  --num-samples "$NUM_SAMPLES" \
  --seed "$SEED" \
  --out "$OUT/a/requests_s${SEED}.npz"

for METHOD in a0 a1 a2; do
  ARGS=(
    --config "$OUT/base.json"
    --ckpt "$OUT/base_s${SEED}/model.pt"
    --out "$OUT/a/${METHOD}_s${SEED}"
    --method "$METHOD"
    --num-samples "$NUM_SAMPLES"
    --batch-size "$BATCH_SIZE"
    --seed "$SEED"
    --requests "$OUT/a/requests_s${SEED}.npz"
    --palette "$OUT/processed/palette.json"
  )
  if [[ "$METHOD" != "a0" ]]; then
    ARGS+=(--candidate-budget 24)
  fi
  if [[ "$METHOD" == "a2" ]]; then
    ARGS+=(--feedback-steps 70,85,95)
  fi
  python -m knit.sample "${ARGS[@]}"
done

for METHOD in a0 a1 a2; do
  python -m knit.evaluate \
    --run-dir "$OUT/a/${METHOD}_s${SEED}" \
    --train-labels "$OUT/processed/train_labels.npy" \
    --out "$OUT/a/${METHOD}_s${SEED}/eval.json"
done

python -m knit.evaluate \
  --compare \
  "$OUT/a/a0_s${SEED}" \
  "$OUT/a/a1_s${SEED}" \
  "$OUT/a/a2_s${SEED}" \
  --train-labels "$OUT/processed/train_labels.npy" \
  --out "$OUT/a/compare_s${SEED}.json"

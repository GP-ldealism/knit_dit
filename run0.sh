#!/bin/bash
#SBATCH --job-name=knit-prep
#SBATCH --time=01:00:00
#SBATCH --output=knit-prep-%j.out
#SBATCH --error=knit-prep-%j.err

set -euo pipefail

# One-off preparation: audit/split/encode + server configs + shared requests.
# Submit this once before the method-screening jobs:
#   sbatch --gpus=1 run0.sh
export REQUEST_SAMPLES="${REQUEST_SAMPLES:-1000}"

PROJECT="${PROJECT:-/data02/run01/scv7se2/knit_dit}"
exec bash "$PROJECT/run.sh" prep

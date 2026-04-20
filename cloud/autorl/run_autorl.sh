#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python -m pip install --upgrade pip
python -m pip install -r cloud/autorl/requirements.txt

python scripts/train_torch_world_model.py --config configs/pytorch_world_model_autorl.json

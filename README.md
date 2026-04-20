# World Model Semantic

This repository contains two related research code paths:

1. A PyTorch world-model reinforcement learning pipeline for semantic task offloading in device-edge networks.
2. A DeepSC-based semantic communication pipeline for training and evaluating semantic transmission quality with BLEU and BERT metrics.

The codebase was organized for local development and remote server training. Large datasets, checkpoints, and training outputs are intentionally excluded from Git tracking.

## Repository Structure

- `configs/`: configuration files for the semantic environment and world-model training.
- `envs/`: custom semantic task offloading environment.
- `scripts/`: training, evaluation, and plotting scripts for the world-model pipeline.
- `deepsc/`: DeepSC training, preprocessing, and semantic performance evaluation code.
- `dreamerv3-main/`: reference DreamerV3 code retained for comparison and adaptation.
- `paper/`: notes and paper-related local materials.

## Main Entry Points

### 1. World-Model Training

Train the task-offloading world model:

```bash
python scripts/train_torch_world_model.py --config configs/pytorch_world_model_autorl.json
```

Useful alternative configs:

- `configs/pytorch_world_model_smoke_100.json`: small smoke test.
- `configs/pytorch_world_model_formal_200.json`: short formal-run estimate.
- `configs/pytorch_world_model_autorl_resume_30000.json`: resume-style config template.

This training script prints:

- current step and total steps
- progress percentage
- elapsed time
- estimated remaining time
- periodic losses and evaluation reward

Typical outputs are written under `outputs/` or remote output directories configured in JSON.

### 2. DeepSC Training

Train the semantic communication model:

```bash
python deepsc/main.py --data-root deepsc --checkpoint-path deepsc/checkpoints/deepsc-Rayleigh --device cuda:0
```

Important current defaults:

- `d_model = 256`
- `channel_symbols = 32`

Checkpoints are written to the directory passed by `--checkpoint-path`.

### 3. DeepSC Performance Evaluation

Evaluate BLEU and BERT semantic similarity curves:

```bash
python deepsc/performance.py ^
  --data-root deepsc ^
  --checkpoint-path deepsc/checkpoints/deepsc-Rayleigh ^
  --device cuda:0 ^
  --bert-model path\to\all-MiniLM-L6-v2_export ^
  --output-dir deepsc/outputs/deepsc_performance ^
  --output-name final_rayleigh
```

The evaluation script currently supports:

- offline local BERT model loading
- SNR sweep evaluation
- BLEU and BERT curve export
- limiting evaluation set size for faster remote testing

## Data Preparation

The Europarl raw corpus is not included in GitHub.

Expected workflow:

1. Place the raw Europarl text files under `deepsc/data/`.
2. Run preprocessing:

```bash
python deepsc/preprocess_text.py --input-data-dir deepsc/europarl/en --output-train-dir deepsc/europarl/train_data.pkl --output-test-dir deepsc/europarl/test_data.pkl --output-vocab deepsc/europarl/vocab.json
```

After preprocessing, the DeepSC code expects:

- `deepsc/europarl/train_data.pkl`
- `deepsc/europarl/test_data.pkl`
- `deepsc/europarl/vocab.json`

## Environment Setup

Recommended Python version:

- Python 3.10 to 3.12 for the current PyTorch-based workflow

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

If you need GPU support, install a CUDA-enabled PyTorch build first according to your platform, then install the remaining packages.

## Notes

- The root `.gitignore` excludes datasets, checkpoints, local outputs, and offline language model folders.
- `cloud/` and `autoresearch-master/` are local support directories and are not required for the core training pipeline.
- On Windows, PowerShell commands may use `^` line continuation in copied examples above; on Linux, replace that with `\`.

## Suggested First Runs

For a quick sanity check:

```bash
python scripts/train_torch_world_model.py --config configs/pytorch_world_model_smoke_100.json
```

For formal world-model training:

```bash
python scripts/train_torch_world_model.py --config configs/pytorch_world_model_autorl.json
```

For DeepSC semantic training:

```bash
python deepsc/main.py --data-root deepsc --checkpoint-path deepsc/checkpoints/deepsc-Rayleigh --device cuda:0
```

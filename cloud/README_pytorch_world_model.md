# PyTorch World Model Cloud Run

## Structure

- `configs/pytorch_world_model.json`: training config
- `configs/pytorch_world_model_autorl.json`: AutoRL/cloud formal config
- `configs/semantic_env_formal.json`: semantic system environment
- `envs/semantic_gym_env.py`: standalone environment
- `scripts/train_torch_world_model.py`: training entrypoint
- `cloud/autorl/`: AutoRL upload files

## Recommended Cloud Setup

- Python 3.10 or 3.11
- PyTorch with CUDA
- One GPU with at least 6 GB VRAM
- Recommended for formal runs: RTX 5090 32GB

## Install

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib scipy
```

## Run

```bash
python scripts/train_torch_world_model.py --config configs/pytorch_world_model.json
```

For AutoRL:

```bash
bash cloud/autorl/run_autorl.sh
```

## Outputs

- `outputs/pytorch_world_model/checkpoint_*.pt`
- `outputs/pytorch_world_model/episode_rewards.json`
- `outputs/pytorch_world_model/metrics.json`
- `outputs/pytorch_world_model/reward_curve.png`

## Notes

- Set `"device": "cuda"` in `configs/pytorch_world_model.json` on cloud GPU.
- Increase `total_steps` and `hidden_dim` only after the first successful run.
- Use `configs/pytorch_world_model_autorl.json` for the formal multi-user cloud run.

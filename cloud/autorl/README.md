# AutoRL Upload Package

## Upload These Paths

- `configs/semantic_env_formal.json`
- `configs/pytorch_world_model_autorl.json`
- `envs/semantic_gym_env.py`
- `scripts/train_torch_world_model.py`
- `cloud/autorl/requirements.txt`
- `cloud/autorl/run_autorl.sh`

## Suggested Runtime

- Python 3.10 or 3.11
- CUDA-enabled PyTorch
- 1 x RTX 5090 32GB

## Start Command

```bash
bash cloud/autorl/run_autorl.sh
```

## Main Outputs

- `outputs/pytorch_world_model_autorl/checkpoint_*.pt`
- `outputs/pytorch_world_model_autorl/episode_rewards.json`
- `outputs/pytorch_world_model_autorl/metrics.json`
- `outputs/pytorch_world_model_autorl/reward_curve.png`

## Notes

- The environment is the multi-terminal multi-edge version with fixed `5` edge servers and `10-40` active users per slot.
- Delay and energy normalizers are scaled for the current `1-2 Mbit` task size setting.
- If AutoRL already provides CUDA PyTorch, you can skip the install step in `run_autorl.sh` and run the training command directly.
- The formal config enables AMP, `torch.compile`, and multiple gradient updates per environment step to improve GPU utilization on large cards.

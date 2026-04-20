import argparse
import datetime
import pathlib
import subprocess
import sys


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--config',
      default='configs/semantic_env.json',
      help='Path to semantic environment JSON config.')
  parser.add_argument(
      '--logdir',
      default='outputs/dreamer_semantic',
      help='Directory for DreamerV3 logs.')
  parser.add_argument('--steps', type=int, default=20000)
  parser.add_argument('--train-ratio', type=float, default=32.0)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--platform', default='cpu')
  parser.add_argument('--envs', type=int, default=None)
  parser.add_argument('--eval-envs', type=int, default=None)
  parser.add_argument('--batch-size', type=int, default=None)
  parser.add_argument('--batch-length', type=int, default=None)
  parser.add_argument('--report-every', type=int, default=None)
  parser.add_argument('--log-every', type=int, default=None)
  parser.add_argument('--save-every', type=int, default=None)
  parser.add_argument('--task', default='semantic_default')
  args = parser.parse_args()

  root = pathlib.Path(__file__).resolve().parents[1]
  config_path = (root / args.config).resolve()
  logdir = (root / args.logdir).resolve()
  if (logdir / 'ckpt').exists():
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logdir = logdir.parent / f'{logdir.name}_{stamp}'
  logdir.mkdir(parents=True, exist_ok=True)
  configs = ['semantic']
  if args.debug:
    configs.append('debug')

  cmd = [
      sys.executable,
      str(root / 'scripts' / 'run_dreamer_main.py'),
      '--logdir', str(logdir),
      '--configs', *configs,
      '--task', args.task,
      '--seed', str(args.seed),
      '--run.steps', str(args.steps),
      '--run.train_ratio', str(args.train_ratio),
      '--jax.platform', args.platform,
      '--env.semantic.config_path', str(config_path),
  ]
  optional = {
      '--run.envs': args.envs,
      '--run.eval_envs': args.eval_envs,
      '--batch_size': args.batch_size,
      '--batch_length': args.batch_length,
      '--run.report_every': args.report_every,
      '--run.log_every': args.log_every,
      '--run.save_every': args.save_every,
  }
  for key, value in optional.items():
    if value is not None:
      cmd.extend([key, str(value)])
  subprocess.run(cmd, check=True, cwd=root)


if __name__ == '__main__':
  main()

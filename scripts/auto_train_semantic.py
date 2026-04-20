import argparse
import json
import pathlib
import subprocess
import sys


def parse_score(logdir):
  score_file = logdir / 'scores.jsonl'
  if not score_file.exists():
    return None
  last = None
  with score_file.open('r', encoding='utf-8') as f:
    for line in f:
      line = line.strip()
      if line:
        last = json.loads(line)
  return None if last is None else last.get('episode/score')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--plan',
      default='configs/auto_train.json',
      help='Batch training plan file.')
  args = parser.parse_args()

  root = pathlib.Path(__file__).resolve().parents[1]
  with (root / args.plan).open('r', encoding='utf-8') as f:
    plan = json.load(f)

  results = []
  for run in plan['runs']:
    logdir = root / 'outputs' / run['name']
    cmd = [
        sys.executable,
        str(root / 'scripts' / 'train_dreamer_semantic.py'),
        '--config', run['config'],
        '--logdir', str(logdir.relative_to(root)),
        '--steps', str(run['steps']),
        '--train-ratio', str(run['train_ratio']),
        '--seed', str(run['seed']),
    ]
    optional = {
        '--platform': run.get('platform'),
        '--task': run.get('task'),
        '--envs': run.get('envs'),
        '--eval-envs': run.get('eval_envs'),
        '--batch-size': run.get('batch_size'),
        '--batch-length': run.get('batch_length'),
        '--report-every': run.get('report_every'),
        '--log-every': run.get('log_every'),
        '--save-every': run.get('save_every'),
    }
    for key, value in optional.items():
      if value is not None:
        cmd.extend([key, str(value)])
    if run.get('debug', False):
      cmd.append('--debug')
    subprocess.run(cmd, check=True, cwd=root)
    results.append({
        'name': run['name'],
        'score': parse_score(logdir),
        'logdir': str(logdir),
    })

  result_file = root / 'outputs' / 'auto_train_results.json'
  with result_file.open('w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)


if __name__ == '__main__':
  main()

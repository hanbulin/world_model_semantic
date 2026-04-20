import argparse
import json
import pathlib

import matplotlib.pyplot as plt


def load_scores(path):
  episodes, rewards = [], []
  with path.open('r', encoding='utf-8') as f:
    for idx, line in enumerate(f, 1):
      line = line.strip()
      if not line:
        continue
      item = json.loads(line)
      reward = item.get('episode/score')
      if reward is not None:
        episodes.append(idx)
        rewards.append(reward)
  return episodes, rewards


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--logdir', required=True)
  args = parser.parse_args()

  root = pathlib.Path(__file__).resolve().parents[1]
  logdir = pathlib.Path(args.logdir)
  if not logdir.is_absolute():
    logdir = root / logdir
  score_file = logdir / 'scores.jsonl'
  episodes, rewards = load_scores(score_file)
  if not rewards:
    raise RuntimeError(f'No episode/score found in {score_file}')

  plt.figure(figsize=(8, 5))
  plt.plot(episodes, rewards, color='tab:blue', linewidth=1.5)
  plt.xlabel('Episode index')
  plt.ylabel('Episode reward')
  plt.title('Reward convergence')
  plt.grid(True, alpha=0.3)
  out = logdir / 'reward_curve.png'
  plt.tight_layout()
  plt.savefig(out, dpi=200)
  print(out)


if __name__ == '__main__':
  main()

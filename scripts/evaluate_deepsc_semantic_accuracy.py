import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEEPSC_ROOT = ROOT / 'deepsc'
if str(DEEPSC_ROOT) not in sys.path:
  sys.path.insert(0, str(DEEPSC_ROOT))

from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from utils import BleuScore, SNR_to_noise, SeqtoText, greedy_decode


def parse_args():
  parser = argparse.ArgumentParser(
      description='Evaluate DeepSC semantic accuracy curves on a remote server.')
  parser.add_argument(
      '--checkpoint',
      action='append',
      required=True,
      help=(
          'Checkpoint spec. Use either '
          '"label=relative/or/absolute/checkpoint.pth" or just a checkpoint path. '
          'Repeat this argument to plot multiple curves in one figure.'))
  parser.add_argument('--channel', default='Rayleigh', choices=['AWGN', 'Rayleigh', 'Rician'])
  parser.add_argument('--vocab-file', default='deepsc/europarl/vocab.json')
  parser.add_argument('--split', default='test', choices=['train', 'test'])
  parser.add_argument('--batch-size', type=int, default=64)
  parser.add_argument('--max-length', type=int, default=30)
  parser.add_argument('--d-model', type=int, default=128)
  parser.add_argument('--dff', type=int, default=512)
  parser.add_argument('--num-layers', type=int, default=4)
  parser.add_argument('--num-heads', type=int, default=8)
  parser.add_argument('--snr-start', type=float, default=-6.0)
  parser.add_argument('--snr-stop', type=float, default=18.0)
  parser.add_argument('--snr-step', type=float, default=3.0)
  parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
  parser.add_argument('--output-dir', default='outputs/deepsc_semantic_accuracy')
  parser.add_argument('--output-name', default='deepsc_semantic_accuracy')
  return parser.parse_args()


def resolve_path(path_str):
  path = pathlib.Path(path_str)
  if path.is_absolute():
    return path
  return ROOT / path


def parse_checkpoint_specs(specs):
  parsed = []
  for spec in specs:
    if '=' in spec:
      label, path_str = spec.split('=', 1)
    else:
      path_str = spec
      label = pathlib.Path(path_str).stem
    parsed.append((label.strip(), resolve_path(path_str.strip())))
  return parsed


def build_model(args, num_vocab, device):
  model = DeepSC(
      args.num_layers,
      num_vocab,
      num_vocab,
      num_vocab,
      num_vocab,
      args.d_model,
      args.num_heads,
      args.dff,
      0.1).to(device)
  return model


def load_vocab(vocab_file):
  vocab = json.loads(resolve_path(vocab_file).read_text(encoding='utf-8'))
  token_to_idx = vocab['token_to_idx']
  idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
  return token_to_idx, idx_to_token


def build_dataloader(split, batch_size):
  dataset = EurDataset(split)
  return DataLoader(
      dataset,
      batch_size=batch_size,
      num_workers=0,
      pin_memory=True,
      collate_fn=collate_data)


def evaluate_curve(model, dataloader, snr_values, channel, max_length, token_to_idx, idx_to_token, device):
  bleu_score = BleuScore(1, 0, 0, 0)
  end_idx = token_to_idx['<END>']
  start_idx = token_to_idx['<START>']
  pad_idx = token_to_idx['<PAD>']
  seq_to_text = SeqtoText(token_to_idx, end_idx)

  model.eval()
  scores = []
  with torch.no_grad():
    for snr in tqdm(snr_values, desc='SNR sweep', leave=False):
      tx_word = []
      rx_word = []
      noise_std = SNR_to_noise(snr)
      for sents in dataloader:
        sents = sents.to(device)
        out = greedy_decode(
            model,
            sents,
            noise_std,
            max_length,
            pad_idx,
            start_idx,
            channel)
        decoded = out.cpu().numpy().tolist()
        target = sents.cpu().numpy().tolist()
        tx_word.extend(map(seq_to_text.sequence_to_text, decoded))
        rx_word.extend(map(seq_to_text.sequence_to_text, target))
      per_sentence_scores = bleu_score.compute_blue_score(tx_word, rx_word)
      scores.append(float(np.mean(per_sentence_scores)))
  return scores


def save_outputs(output_dir, output_name, snr_values, series_results):
  output_dir.mkdir(parents=True, exist_ok=True)
  json_path = output_dir / f'{output_name}.json'
  png_path = output_dir / f'{output_name}.png'

  payload = {
      'snr_db': list(map(float, snr_values)),
      'series': [
          {'label': label, 'semantic_accuracy': list(map(float, values))}
          for label, values in series_results
      ],
  }
  json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

  plt.figure(figsize=(8, 5))
  for label, values in series_results:
    plt.plot(snr_values, values, marker='o', linewidth=2, label=label)
  plt.xlabel('SNR (dB)')
  plt.ylabel('Semantic accuracy (BLEU-1)')
  plt.title('DeepSC semantic accuracy curves')
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(png_path, dpi=200)
  plt.close()
  return json_path, png_path


def main():
  args = parse_args()
  device = torch.device(args.device)
  snr_values = np.arange(args.snr_start, args.snr_stop + 1e-9, args.snr_step, dtype=np.float32)
  token_to_idx, idx_to_token = load_vocab(args.vocab_file)
  num_vocab = len(token_to_idx)
  dataloader = build_dataloader(args.split, args.batch_size)
  checkpoint_specs = parse_checkpoint_specs(args.checkpoint)

  series_results = []
  for label, ckpt_path in checkpoint_specs:
    if not ckpt_path.exists():
      raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')
    model = build_model(args, num_vocab, device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    values = evaluate_curve(
        model,
        dataloader,
        snr_values,
        args.channel,
        args.max_length,
        token_to_idx,
        idx_to_token,
        device)
    series_results.append((label, values))

  output_dir = resolve_path(args.output_dir)
  json_path, png_path = save_outputs(output_dir, args.output_name, snr_values, series_results)
  print(f'saved semantic accuracy json to {json_path}')
  print(f'saved semantic accuracy figure to {png_path}')


if __name__ == '__main__':
  main()

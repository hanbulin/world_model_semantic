# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
Evaluate DeepSC semantic accuracy curves with BLEU and BERT similarity.
"""

import argparse
import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EurDataset, collate_data, resolve_data_path
from models.transceiver import DeepSC
from utils import BleuScore, SNR_to_noise, SeqtoText, greedy_decode

try:
  from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
  SentenceTransformer = None


parser = argparse.ArgumentParser()
parser.add_argument('--data-root', default='.', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--checkpoint-file', default='', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=256, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--channel-symbols', default=32, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', type=str)
parser.add_argument('--snr-start', default=0.0, type=float)
parser.add_argument('--snr-stop', default=18.0, type=float)
parser.add_argument('--snr-step', default=3.0, type=float)
parser.add_argument('--max-eval-samples', default=5000, type=int)
parser.add_argument('--eval-seed', default=0, type=int)
parser.add_argument('--bert-model', default='sentence-transformers/all-MiniLM-L6-v2', type=str)
parser.add_argument('--disable-bert', action='store_true')
parser.add_argument('--output-dir', default='outputs/deepsc_performance', type=str)
parser.add_argument('--output-name', default='deepsc_semantic_accuracy', type=str)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BertSimilarity:

  def __init__(self, model_name, device_name):
    if SentenceTransformer is None:
      raise ImportError(
          'sentence-transformers is required for BERT similarity. '
          'Install it with `pip install sentence-transformers`.')
    self.model = SentenceTransformer(model_name, device=device_name)

  def compute_similarity(self, real, predicted):
    emb_real = self.model.encode(real, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    emb_pred = self.model.encode(predicted, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return np.sum(emb_real * emb_pred, axis=1).tolist()


def resolve_checkpoint_file(checkpoint_path, checkpoint_file):
  if checkpoint_file:
    path = pathlib.Path(checkpoint_file)
    if not path.is_absolute():
      path = pathlib.Path(checkpoint_path) / checkpoint_file
    return path

  checkpoint_dir = pathlib.Path(checkpoint_path)
  if (checkpoint_dir / 'checkpoint_best.pth').exists():
    return checkpoint_dir / 'checkpoint_best.pth'
  if (checkpoint_dir / 'checkpoint_last.pth').exists():
    return checkpoint_dir / 'checkpoint_last.pth'

  model_paths = []
  for fn in os.listdir(checkpoint_dir):
    if not fn.endswith('.pth'):
      continue
    stem = pathlib.Path(fn).stem
    if stem.startswith('checkpoint_') and stem.split('_')[-1].isdigit():
      idx = int(stem.split('_')[-1])
      model_paths.append((checkpoint_dir / fn, idx))
  if not model_paths:
    raise FileNotFoundError(f'No checkpoint found in {checkpoint_dir}')
  model_paths.sort(key=lambda x: x[1])
  return model_paths[-1][0]


def performance(args, snr_values, net, token_to_idx, idx_to_token):
  bleu_score_1gram = BleuScore(1, 0, 0, 0)
  bert_similarity = None if args.disable_bert else BertSimilarity(args.bert_model, args.device)

  test_eur = EurDataset('test', data_root=args.data_root)
  if args.max_eval_samples and args.max_eval_samples < len(test_eur):
    rng = np.random.default_rng(args.eval_seed)
    indices = rng.choice(len(test_eur), size=args.max_eval_samples, replace=False).tolist()
    test_eur = torch.utils.data.Subset(test_eur, indices)
  test_iterator = DataLoader(
      test_eur,
      batch_size=args.batch_size,
      num_workers=0,
      pin_memory=True,
      collate_fn=collate_data)

  end_idx = token_to_idx['<END>']
  start_idx = token_to_idx['<START>']
  pad_idx = token_to_idx['<PAD>']
  seq_to_text = SeqtoText(token_to_idx, end_idx)

  bleu_scores = []
  bert_scores = []
  net.eval()
  with torch.no_grad():
    for snr in tqdm(snr_values, desc='SNR sweep'):
      predicted_sentences = []
      target_sentences = []
      noise_std = SNR_to_noise(snr)

      for sents in test_iterator:
        sents = sents.to(device)
        out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx, start_idx, args.channel)

        decoded = out.cpu().numpy().tolist()
        target = sents.cpu().numpy().tolist()
        predicted_sentences.extend(map(seq_to_text.sequence_to_text, decoded))
        target_sentences.extend(map(seq_to_text.sequence_to_text, target))

      bleu_values = bleu_score_1gram.compute_blue_score(predicted_sentences, target_sentences)
      bleu_scores.append(float(np.mean(bleu_values)))

      if bert_similarity is not None:
        bert_values = bert_similarity.compute_similarity(predicted_sentences, target_sentences)
        bert_scores.append(float(np.mean(bert_values)))

  return bleu_scores, bert_scores, len(test_eur)


def plot_curve(x, y, xlabel, ylabel, title, out_path):
  plt.figure(figsize=(8, 5))
  plt.plot(x, y, marker='o', linewidth=2)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.savefig(out_path, dpi=200)
  plt.close()


if __name__ == '__main__':
  args = parser.parse_args()
  args.data_root = str(resolve_data_path(args.data_root))
  args.vocab_file = str(resolve_data_path(args.vocab_file))
  args.checkpoint_path = str(resolve_data_path(args.checkpoint_path))
  args.output_dir = str(resolve_data_path(args.output_dir))

  device = torch.device(args.device)
  snr_values = np.arange(args.snr_start, args.snr_stop + 1e-9, args.snr_step, dtype=np.float32)

  vocab = json.loads(pathlib.Path(args.vocab_file).read_text(encoding='utf-8'))
  token_to_idx = vocab['token_to_idx']
  idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
  num_vocab = len(token_to_idx)

  deepsc = DeepSC(
      args.num_layers,
      num_vocab,
      num_vocab,
      num_vocab,
      num_vocab,
      args.d_model,
      args.num_heads,
      args.dff,
      0.1,
      channel_symbols=args.channel_symbols).to(device)

  checkpoint_file = resolve_checkpoint_file(args.checkpoint_path, args.checkpoint_file)
  checkpoint = torch.load(checkpoint_file, map_location=device)
  deepsc.load_state_dict(checkpoint)
  print(f'model loaded from {checkpoint_file}')

  bleu_scores, bert_scores, num_eval_samples = performance(
      args, snr_values, deepsc, token_to_idx, idx_to_token)

  output_dir = pathlib.Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  metrics = {
      'checkpoint': str(checkpoint_file),
      'channel': args.channel,
      'num_eval_samples': int(num_eval_samples),
      'snr_db': list(map(float, snr_values)),
      'bleu_1gram': list(map(float, bleu_scores)),
  }
  if bert_scores:
    metrics['bert_similarity'] = list(map(float, bert_scores))

  json_path = output_dir / f'{args.output_name}.json'
  json_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

  bleu_png = output_dir / f'{args.output_name}_bleu.png'
  plot_curve(
      snr_values,
      bleu_scores,
      'SNR (dB)',
      'BLEU-1',
      'DeepSC BLEU semantic accuracy curve',
      bleu_png)

  print(f'saved BLEU curve to {bleu_png}')
  if bert_scores:
    bert_png = output_dir / f'{args.output_name}_bert.png'
    plot_curve(
        snr_values,
        bert_scores,
        'SNR (dB)',
        'BERT cosine similarity',
        'DeepSC BERT semantic accuracy curve',
        bert_png)
    print(f'saved BERT curve to {bert_png}')
  print(f'saved metrics to {json_path}')

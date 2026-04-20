# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
Dataset utilities for DeepSC.
"""

import os
import pickle
import pathlib

import numpy as np
import torch
from torch.utils.data import Dataset


def _default_data_root():
  env_root = os.environ.get('DEEPSC_DATA_ROOT')
  if env_root:
    return pathlib.Path(env_root)
  return pathlib.Path(__file__).resolve().parent


def resolve_data_path(relative_path):
  path = pathlib.Path(relative_path)
  if path.is_absolute():
    return path
  return _default_data_root() / path


class EurDataset(Dataset):

  def __init__(self, split='train', data_root=None):
    root = pathlib.Path(data_root) if data_root is not None else _default_data_root()
    data_path = root / 'europarl' / f'{split}_data.pkl'
    with data_path.open('rb') as f:
      self.data = pickle.load(f)

  def __getitem__(self, index):
    return self.data[index]

  def __len__(self):
    return len(self.data)


def collate_data(batch):
  batch_size = len(batch)
  max_len = max(map(len, batch))
  sents = np.zeros((batch_size, max_len), dtype=np.int64)
  sort_by_len = sorted(batch, key=len, reverse=True)

  for i, sent in enumerate(sort_by_len):
    length = len(sent)
    sents[i, :length] = sent

  return torch.from_numpy(sents)

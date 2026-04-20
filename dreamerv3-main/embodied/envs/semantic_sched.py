import json
import pathlib

import elements
import embodied
import numpy as np


class SemanticSched(embodied.Env):

  def __init__(self, task, config_path='', seed=0, **kwargs):
    del task, kwargs
    self._rng = np.random.default_rng(seed)
    self._config = self._load_config(config_path)
    self._episode_length = int(self._config['episode_length'])
    self._num_edges = int(self._config['num_edges'])
    self._edge_load = np.zeros(self._num_edges, np.float32)
    self._device_energy = float(self._config['device_energy_budget'])
    self._step_count = 0
    self._done = True
    self._task = {}

  @property
  def obs_space(self):
    size = 7 + self._num_edges
    return {
        'vector': elements.Space(np.float32, (size,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'reset': elements.Space(bool),
        'mode': elements.Space(np.int32, (), 0, 3),
        'semantic_level': elements.Space(
            np.int32, (), 0, len(self._config['semantic_levels'])),
        'power_level': elements.Space(
            np.int32, (), 0, len(self._config['power_levels'])),
        'local_cpu_level': elements.Space(
            np.int32, (), 0, len(self._config['local_cpu_levels'])),
        'edge_cpu_level': elements.Space(
            np.int32, (), 0, len(self._config['edge_cpu_levels'])),
        'e2e_rate_level': elements.Space(
            np.int32, (), 0, len(self._config['e2e_rate_levels'])),
        'target_edge': elements.Space(np.int32, (), 0, self._num_edges),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._step_count = 0
      self._done = False
      self._edge_load = np.zeros(self._num_edges, np.float32)
      self._device_energy = float(self._config['device_energy_budget'])
      self._task = self._sample_task()
      return self._obs(0.0, is_first=True)

    metrics = self._transition(action)
    self._step_count += 1
    self._done = (
        self._step_count >= self._episode_length or self._device_energy <= 0.0)
    self._task = self._sample_task()
    return self._obs(
        metrics['reward'],
        is_last=self._done,
        is_terminal=self._done)

  def _load_config(self, config_path):
    if not config_path:
      config_path = (
          pathlib.Path(__file__).resolve().parents[3] /
          'configs' / 'semantic_env.json')
    path = pathlib.Path(config_path)
    with path.open('r', encoding='utf-8') as f:
      return json.load(f)

  def _sample_task(self):
    task = {
        'data_size': float(self._rng.uniform(*self._config['data_size_range'])),
        'workload': float(self._rng.uniform(*self._config['workload_range'])),
        'sentence_len': float(
            self._rng.integers(*self._config['sentence_len_range'])),
        'snr_d2e_db': float(self._rng.uniform(*self._config['snr_d2e_db_range'])),
        'snr_e2e_db': float(self._rng.uniform(*self._config['snr_e2e_db_range'])),
    }
    task['affiliated_edge'] = int(self._rng.integers(0, self._num_edges))
    return task

  def _obs(
      self, reward, is_first=False, is_last=False, is_terminal=False):
    vec = np.array([
        self._task['data_size'] / self._config['data_size_range'][1],
        self._task['workload'] / self._config['workload_range'][1],
        self._task['sentence_len'] / self._config['sentence_len_range'][1],
        self._task['snr_d2e_db'] / self._config['snr_d2e_db_range'][1],
        self._task['snr_e2e_db'] / self._config['snr_e2e_db_range'][1],
        self._device_energy / self._config['device_energy_budget'],
        self._task['affiliated_edge'] / max(1, self._num_edges - 1),
        *self._edge_load.tolist(),
    ], np.float32)
    return {
        'vector': vec,
        'reward': np.float32(reward),
        'is_first': is_first,
        'is_last': is_last,
        'is_terminal': is_terminal,
    }

  def _transition(self, action):
    mode = int(action['mode'])
    level = self._config['semantic_levels'][int(action['semantic_level'])]
    tx_power = self._config['power_levels'][int(action['power_level'])]
    local_cpu = self._config['local_cpu_levels'][int(action['local_cpu_level'])]
    edge_cpu = self._config['edge_cpu_levels'][int(action['edge_cpu_level'])]
    e2e_rate = self._config['e2e_rate_levels'][int(action['e2e_rate_level'])]
    target_edge = int(action['target_edge']) % self._num_edges

    affiliated = self._task['affiliated_edge']
    local_delay = self._task['workload'] / local_cpu
    local_energy = (
        self._config['alpha_local'] * self._task['workload'] * local_cpu ** 2)

    if mode == 0:
      delay = local_delay
      energy = local_energy
      accuracy = 1.0
      edge_index = None
    else:
      compressed = level['eta'] * self._task['sentence_len']
      accuracy = self._semantic_accuracy(level, self._task['snr_d2e_db'])
      delay, energy, edge_index = self._offload_metrics(
          mode, affiliated, target_edge, tx_power, local_cpu, edge_cpu,
          e2e_rate, compressed)

    semantic_penalty = max(
        0.0, self._config['semantic_requirement'] - accuracy)
    energy_penalty = max(0.0, energy - self._device_energy)
    reward = (
        -self._config['reward_weights']['delay'] *
        (delay / self._config['delay_normalizer'])
        -self._config['reward_weights']['energy'] *
        (energy / self._config['energy_normalizer'])
        +self._config['reward_weights']['semantic'] * accuracy
        -self._config['reward_weights']['penalty'] *
        (semantic_penalty + energy_penalty))

    self._device_energy = max(0.0, self._device_energy - energy)
    self._edge_load *= self._config['edge_load_decay']
    if edge_index is not None:
      self._edge_load[edge_index] = min(
          1.0,
          self._edge_load[edge_index] +
          edge_cpu / self._config['edge_cpu_levels'][-1])
    return {
        'delay': delay,
        'energy': energy,
        'accuracy': accuracy,
        'reward': reward,
    }

  def _offload_metrics(
      self, mode, affiliated, target_edge, tx_power, local_cpu, edge_cpu,
      e2e_rate, compressed):
    snr_d2e = self._db_to_linear(self._task['snr_d2e_db'])
    bandwidth = self._config['bandwidth_d2e']
    d2e_rate = bandwidth * np.log2(1.0 + tx_power * snr_d2e)
    d2e_rate = max(d2e_rate, 1e-6)

    encode_delay = self._task['data_size'] / local_cpu
    d2e_delay = compressed / d2e_rate
    decode_delay = compressed / edge_cpu
    exec_delay = self._task['workload'] / edge_cpu

    encode_energy = (
        self._config['alpha_encode'] * self._task['data_size'] * local_cpu ** 2)
    d2e_energy = tx_power * d2e_delay
    decode_energy = (
        self._config['alpha_decode'] * compressed * edge_cpu ** 2)
    exec_energy = (
        self._config['alpha_edge'] * self._task['workload'] * edge_cpu ** 2)

    total_delay = encode_delay + d2e_delay + decode_delay + exec_delay
    total_energy = encode_energy + d2e_energy + decode_energy + exec_energy
    edge_index = affiliated

    if mode == 2:
      snr_e2e = self._db_to_linear(self._task['snr_e2e_db'])
      remote_rate = e2e_rate * np.log2(1.0 + snr_e2e)
      remote_rate = max(remote_rate, 1e-6)
      total_delay += compressed / remote_rate
      total_energy += self._config['edge_forward_power'] * compressed / remote_rate
      edge_index = target_edge

    return total_delay, total_energy, edge_index

  def _semantic_accuracy(self, level, snr_db):
    accuracy = (
        level['A'] /
        (1.0 + np.exp(-level['B'] * (snr_db - level['C']))) +
        level['D'])
    return float(np.clip(accuracy, 0.0, 1.0))

  def _db_to_linear(self, value_db):
    return 10.0 ** (value_db / 10.0)

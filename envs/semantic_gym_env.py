import json
import math
import pathlib

import numpy as np
import torch


class SemanticGymEnv:

  def __init__(self, config_path, seed=0):
    self._rng = np.random.default_rng(seed)
    self._config = self._load_config(config_path)
    self._torch_device = self._select_torch_device(
        self._config.get('env_compute_device', 'auto'))
    self._episode_length = int(self._config['episode_length'])
    self._num_edges = int(self._config['num_edges'])
    self._num_users_range = tuple(self._config.get('num_users_range', [4, 6]))
    self._max_users = int(self._num_users_range[1])
    self._user_feature_dim = 11
    per_user_action = [
        3,
        len(self._config['semantic_levels']),
        self._num_edges,
    ]
    self._action_dims = per_user_action * self._max_users
    self.obs_dim = self._max_users * self._user_feature_dim + self._num_edges
    self.action_dim = sum(self._action_dims)
    self._edge_load = np.zeros(self._num_edges, np.float32)
    self._step_count = 0
    self._tasks = []
    self._active_users = self._max_users
    self._device_cpu_totals = np.zeros(self._max_users, np.float32)
    self._edge_cpu_totals = np.zeros(self._num_edges, np.float32)

  @property
  def action_dims(self):
    return list(self._action_dims)

  def reset(self):
    self._step_count = 0
    self._edge_load = np.zeros(self._num_edges, np.float32)
    self._sample_slot_state()
    return self._obs()

  def step(self, action):
    metrics = self._transition(action)
    current_active_users = metrics['active_users']
    current_edge_cpu_totals = metrics['edge_cpu_totals']
    current_device_cpu_totals = metrics['device_cpu_totals']
    self._step_count += 1
    done = self._step_count >= self._episode_length
    obs = self._obs() if done else self._advance_tasks()
    info = {
        'active_users': current_active_users,
        'avg_delay': metrics['avg_delay'],
        'avg_energy': metrics['avg_energy'],
        'avg_accuracy': metrics['avg_accuracy'],
        'total_delay': metrics['total_delay'],
        'total_energy': metrics['total_energy'],
        'reward': metrics['reward'],
        'per_user': metrics['per_user'],
        'edge_cpu_totals': current_edge_cpu_totals,
        'device_cpu_totals': current_device_cpu_totals,
    }
    return obs, float(metrics['reward']), done, info

  def sample_random_action(self):
    return np.array([
        self._rng.integers(0, size) for size in self._action_dims
    ], dtype=np.int64)

  def _advance_tasks(self):
    self._sample_slot_state()
    return self._obs()

  def _sample_slot_state(self):
    self._active_users = int(self._rng.integers(self._num_users_range[0], self._num_users_range[1] + 1))
    self._tasks = [self._sample_task(user_id) for user_id in range(self._active_users)]
    self._device_cpu_totals = np.zeros(self._max_users, np.float32)
    for idx in range(self._active_users):
      bounds = self._config['device_cpu_total_bounds']
      self._device_cpu_totals[idx] = float(self._rng.uniform(bounds[0], bounds[1]))
    bounds = self._config['edge_cpu_total_bounds']
    self._edge_cpu_totals = self._rng.uniform(bounds[0], bounds[1], size=self._num_edges).astype(np.float32)

  def _load_config(self, config_path):
    path = pathlib.Path(config_path)
    if not path.is_absolute():
      path = pathlib.Path(__file__).resolve().parents[1] / config_path
    with path.open('r', encoding='utf-8') as f:
      return json.load(f)

  def _select_torch_device(self, name):
    if name == 'auto':
      return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)

  def _sample_task(self, user_id):
    sentence_len = float(self._rng.integers(*self._config['sentence_len_range']))
    return {
        'user_id': int(user_id),
        'data_size': sentence_len * float(self._config['source_token_bits']),
        'workload': float(self._rng.uniform(*self._config['workload_range'])),
        'sentence_len': sentence_len,
        'channel_gain_d2e': float(self._rng.uniform(*self._config['channel_gain_d2e_range'])),
        'channel_gain_e2e': float(self._rng.uniform(*self._config['channel_gain_e2e_range'])),
        'interference_d2e_power': float(self._rng.uniform(*self._config['interference_d2e_power_range'])),
        'interference_e2e_power': float(self._rng.uniform(*self._config['interference_e2e_power_range'])),
        'user_weight': float(self._rng.uniform(*self._config['user_weight_range'])),
        'energy_weight': float(self._rng.uniform(*self._config['energy_weight_range'])),
        'affiliated_edge': int(self._rng.integers(0, self._num_edges)),
    }

  def _obs(self):
    values = []
    max_bits = self._config['sentence_len_range'][1] * self._config['source_token_bits']
    max_device_cpu = self._config['device_cpu_total_bounds'][1]
    max_edge_cpu = self._config['edge_cpu_total_bounds'][1]
    for idx in range(self._max_users):
      if idx < self._active_users:
        task = self._tasks[idx]
        values.extend([
            1.0,
            task['data_size'] / max_bits,
            task['workload'] / self._config['workload_range'][1],
            task['sentence_len'] / self._config['sentence_len_range'][1],
            task['channel_gain_d2e'] / self._config['channel_gain_d2e_range'][1],
            task['channel_gain_e2e'] / self._config['channel_gain_e2e_range'][1],
            task['interference_d2e_power'] / self._config['interference_d2e_power_range'][1],
            task['interference_e2e_power'] / self._config['interference_e2e_power_range'][1],
            task['user_weight'] / self._config['user_weight_range'][1],
            task['energy_weight'] / self._config['energy_weight_range'][1],
            self._device_cpu_totals[idx] / max_device_cpu,
        ])
      else:
        values.extend([0.0] * self._user_feature_dim)
    values.extend((self._edge_cpu_totals / max_edge_cpu).tolist())
    return np.array(values, dtype=np.float32)

  def _transition(self, action):
    action = np.asarray(action, dtype=np.int64).reshape(self._max_users, 3)
    plans = []
    edge_requests = {edge: [] for edge in range(self._num_edges)}
    d2e_groups = {edge: [] for edge in range(self._num_edges)}
    e2e_groups = {}

    for idx in range(self._active_users):
      task = self._tasks[idx]
      mode, sem_idx, target_edge = [int(x) for x in action[idx].tolist()]
      mode = mode % 3
      level = self._config['semantic_levels'][sem_idx % len(self._config['semantic_levels'])]
      affiliated = task['affiliated_edge']
      exec_edge = affiliated if mode == 1 else target_edge % self._num_edges
      if mode == 2 and exec_edge == affiliated:
        exec_edge = (affiliated + 1) % self._num_edges
      plan = {
          'user_idx': idx,
          'task': task,
          'mode': mode,
          'semantic_level': level,
          'target_edge': None if mode == 0 else int(exec_edge),
          'compressed_bits': 0.0 if mode == 0 else self._compressed_bits(task, level),
          'device_cpu_total': float(self._device_cpu_totals[idx]),
      }
      if mode != 0:
        d2e_groups[affiliated].append(plan)
        edge_requests[exec_edge].append(plan)
        if mode == 2:
          key = (affiliated, exec_edge)
          e2e_groups.setdefault(key, []).append(plan)
      plans.append(plan)

    self._allocate_d2e_resources(d2e_groups)
    self._allocate_edge_resources(edge_requests)
    self._allocate_e2e_resources(e2e_groups)

    per_user_metrics = []
    rewards = []
    total_delay = 0.0
    total_energy = 0.0
    total_accuracy = 0.0

    for plan in plans:
      metrics = self._finalize_user_metrics(plan)
      per_user_metrics.append(metrics)
      rewards.append(metrics['reward'])
      total_delay += metrics['delay']
      total_energy += metrics['energy']
      total_accuracy += metrics['accuracy']
    avg_delay = total_delay / max(self._active_users, 1)
    avg_energy = total_energy / max(self._active_users, 1)
    avg_accuracy = total_accuracy / max(self._active_users, 1)
    reward = float(np.mean(rewards)) if rewards else 0.0
    return {
        'active_users': self._active_users,
        'avg_delay': float(avg_delay),
        'avg_energy': float(avg_energy),
        'avg_accuracy': float(avg_accuracy),
        'total_delay': float(total_delay),
        'total_energy': float(total_energy),
        'reward': reward,
        'per_user': per_user_metrics,
        'edge_cpu_totals': self._edge_cpu_totals.tolist(),
        'device_cpu_totals': self._device_cpu_totals[:self._active_users].tolist(),
    }

  def _allocate_d2e_resources(self, d2e_groups):
    bandwidth_total = float(self._config['bandwidth_d2e'])
    for affiliated, plans in d2e_groups.items():
      if not plans:
        continue
      compressed = torch.tensor(
          [max(plan['compressed_bits'], 1e-6) for plan in plans],
          dtype=torch.float32,
          device=self._torch_device)
      # In the paper system model, D2E bandwidth is a fixed parameter rather than
      # an optimization variable, so each active D2E transmission uses the same
      # configured bandwidth directly.
      bandwidths = torch.full_like(compressed, bandwidth_total)
      powers, snr_dbs, accuracies, feasible_flags = self._solve_d2e_power_closed_form_batch(
          plans, compressed, bandwidths)
      for idx, plan in enumerate(plans):
        bandwidth = float(bandwidths[idx].item())
        plan['bandwidth_d2e'] = float(bandwidth)
        plan['tx_power'] = float(powers[idx])
        plan['snr_d2e_db'] = float(snr_dbs[idx])
        plan['accuracy'] = float(accuracies[idx])
        plan['semantic_feasible'] = bool(feasible_flags[idx])
        plan['d2e_rate'] = float(
            bandwidth * math.log2(1.0 + self._d2e_snr(plan['task'], plan['tx_power'])))

  def _allocate_edge_resources(self, edge_requests):
    decode_min = float(self._config['decode_cpu_min'])
    exec_min = float(self._config['edge_exec_cpu_min'])
    for edge, plans in edge_requests.items():
      if not plans:
        continue
      capacity = self._available_edge_capacity(edge)
      components = []
      for plan in plans:
        components.append({
            'kind': 'decode',
            'plan': plan,
            'a': self._cpu_delay_coeff(plan, 'decode'),
            'b': self._cpu_energy_coeff(plan, 'decode'),
            'lower': decode_min,
            'upper': capacity,
        })
        components.append({
            'kind': 'exec',
            'plan': plan,
            'a': self._cpu_delay_coeff(plan, 'exec'),
            'b': self._cpu_energy_coeff(plan, 'exec'),
            'lower': exec_min,
            'upper': capacity,
        })
      allocations = self._solve_edge_cpu_closed_form(components, capacity)
      for component, alloc in zip(components, allocations):
        if component['kind'] == 'decode':
          component['plan']['f_decode'] = float(max(alloc, 1e-6))
        else:
          component['plan']['f_edge'] = float(max(alloc, 1e-6))

  def _allocate_e2e_resources(self, e2e_groups):
    bounds = self._config['e2e_rate_bounds']
    for (affiliated, target_edge), plans in e2e_groups.items():
      if not plans:
        continue
      rate_total = self._available_e2e_rate_torch(affiliated, target_edge)
      components = [{
          'plan': plan,
          'a': self._e2e_delay_energy_coeff(plan),
          'lower': float(bounds[0]),
          'upper': float(bounds[1]),
      } for plan in plans]
      allocations = self._solve_e2e_rate_closed_form(
          components, min(rate_total, len(components) * float(bounds[1])))
      for component, alloc in zip(components, allocations):
        component['plan']['e2e_rate'] = float(max(alloc, 1e-6))

  def _finalize_user_metrics(self, plan):
    task = plan['task']
    weights = self._effective_weights(task)
    original_bits = float(task['data_size'])
    f_local = 0.0
    f_encode = 0.0
    f_decode = 0.0
    f_edge = 0.0

    if plan['mode'] == 0:
      f_local = float(plan['device_cpu_total'])
      f_encode = 0.0
      delay = task['workload'] / max(f_local, 1e-6)
      energy = self._config['alpha_local'] * task['workload'] * f_local ** 2
      accuracy = 1.0
      compressed_bits = 0.0
      tx_power = 0.0
      d2e_rate = 0.0
      e2e_rate = 0.0
      semantic_level = 'local'
      semantic_dim = 0
      target_edge = None
    else:
      compressed_bits = float(plan['compressed_bits'])
      f_encode = float(plan['device_cpu_total'])
      f_local = 0.0
      encode_delay = task['data_size'] / max(f_encode, 1e-6)
      encode_energy = self._config['alpha_encode'] * task['data_size'] * f_encode ** 2
      d2e_rate = max(plan.get('d2e_rate', 0.0), 1e-6)
      d2e_delay = compressed_bits / d2e_rate
      tx_power = float(plan.get('tx_power', 0.0))
      d2e_energy = tx_power * d2e_delay
      f_decode = max(plan.get('f_decode', 1e-6), 1e-6)
      f_edge = max(plan.get('f_edge', 1e-6), 1e-6)
      decode_delay = compressed_bits / f_decode
      decode_energy = self._config['alpha_decode'] * compressed_bits * f_decode ** 2
      exec_delay = task['workload'] / f_edge
      exec_energy = self._config['alpha_edge'] * task['workload'] * f_edge ** 2
      e2e_rate = float(plan.get('e2e_rate', 0.0))
      e2e_delay = 0.0
      e2e_energy = 0.0
      if plan['mode'] == 2:
        e2e_rate = max(e2e_rate, 1e-6)
        e2e_delay = compressed_bits / e2e_rate
        e2e_energy = self._config['edge_forward_power'] * e2e_delay
      delay = encode_delay + d2e_delay + decode_delay + exec_delay + e2e_delay
      energy = encode_energy + d2e_energy + decode_energy + exec_energy + e2e_energy
      accuracy = float(plan.get('accuracy', 0.0))
      semantic_level = plan['semantic_level']['name']
      semantic_dim = int(plan['semantic_level'].get('k', 0))
      target_edge = plan['target_edge']

    semantic_penalty = max(0.0, self._config['semantic_requirement'] - accuracy)
    reward = (
        -task['user_weight'] * ((delay / self._config['delay_normalizer']) + weights['energy'] * (energy / self._config['energy_normalizer']))
        - weights['penalty'] * semantic_penalty)

    return {
        'user_id': task['user_id'],
        'mode': int(plan['mode']),
        'semantic_level': semantic_level,
        'semantic_dim': semantic_dim,
        'original_bits': original_bits,
        'compressed_bits': float(compressed_bits),
        'delay': float(delay),
        'energy': float(energy),
        'accuracy': float(accuracy),
        'semantic_feasible': bool(accuracy >= self._config['semantic_requirement']),
        'semantic_violation': float(semantic_penalty),
        'reward': float(reward),
        'user_weight': float(task['user_weight']),
        'energy_weight': float(task['energy_weight']),
        'tx_power': float(tx_power),
        'd2e_rate': float(d2e_rate),
        'e2e_rate': float(e2e_rate),
        'f_local': float(f_local),
        'f_encode': float(f_encode),
        'f_decode': float(f_decode),
        'f_edge': float(f_edge),
        'device_cpu_total': float(plan['device_cpu_total']),
        'target_edge': target_edge,
        'objective': float(self._weighted_objective(delay, energy, task)),
    }

  def _optimize_d2e_power_dinkelbach(self, task, level, compressed, bandwidth):
    power_min, power_max = self._config['power_bounds']
    required_power = max(power_min, self._minimum_power_for_semantic(task, level))
    feasible = required_power <= power_max + 1e-9
    if not feasible:
      chosen_power = float(power_max)
      snr_db = self._linear_to_db(self._d2e_snr(task, chosen_power))
      accuracy = self._semantic_accuracy(level, snr_db)
      return chosen_power, float(snr_db), float(accuracy), False

    beta = task['user_weight'] * compressed * task['energy_weight'] / max(
        self._config['energy_normalizer'], 1e-9)
    alpha = task['user_weight'] * compressed / max(
        self._config['delay_normalizer'], 1e-9)
    channel_coeff = task['channel_gain_d2e'] / max(
        self._config['noise_power'] + task['interference_d2e_power'], 1e-18)
    log_scale = bandwidth
    lower = float(required_power)
    upper = float(power_max)

    def rate_fn(power):
      return log_scale * math.log2(1.0 + channel_coeff * power)

    def numerator(power):
      return alpha + beta * power

    def solve_inner(q_value):
      slope = beta
      if q_value <= 0.0 or slope <= 1e-12:
        candidate = lower
      else:
        root = (q_value * log_scale / max(slope * math.log(2.0), 1e-12)) - (1.0 / max(channel_coeff, 1e-18))
        candidate = min(max(root, lower), upper)
      boundary_points = [lower, upper, candidate]
      best_power = lower
      best_value = None
      for point in boundary_points:
        value = numerator(point) - q_value * rate_fn(point)
        if best_value is None or value < best_value:
          best_value = value
          best_power = point
      return float(best_power), float(best_value)

    q_value = numerator(lower) / max(rate_fn(lower), 1e-9)
    chosen_power = lower
    max_iters = int(self._config.get('dinkelbach_max_iters', 40))
    tolerance = float(self._config.get('dinkelbach_tolerance', 1e-6))
    for _ in range(max_iters):
      chosen_power, residual = solve_inner(q_value)
      rate = rate_fn(chosen_power)
      if abs(residual) <= tolerance:
        break
      q_value = numerator(chosen_power) / max(rate, 1e-9)

    snr_db = self._linear_to_db(self._d2e_snr(task, chosen_power))
    accuracy = self._semantic_accuracy(level, snr_db)
    return float(chosen_power), float(snr_db), float(accuracy), True

  def _optimize_d2e_power_dinkelbach_batch(self, plans, compressed, bandwidths):
    device = self._torch_device
    power_min, power_max = self._config['power_bounds']
    noise_power = float(self._config['noise_power'])
    semantic_req = float(self._config['semantic_requirement'])
    delay_norm = float(self._config['delay_normalizer'])
    energy_norm = float(self._config['energy_normalizer'])

    channel_gain = torch.tensor(
        [plan['task']['channel_gain_d2e'] for plan in plans],
        dtype=torch.float32,
        device=device)
    interference = torch.tensor(
        [plan['task']['interference_d2e_power'] for plan in plans],
        dtype=torch.float32,
        device=device)
    user_weight = torch.tensor(
        [plan['task']['user_weight'] for plan in plans],
        dtype=torch.float32,
        device=device)
    energy_weight = torch.tensor(
        [plan['task']['energy_weight'] for plan in plans],
        dtype=torch.float32,
        device=device)
    level_a = torch.tensor(
        [plan['semantic_level']['A'] for plan in plans],
        dtype=torch.float32,
        device=device)
    level_b = torch.tensor(
        [plan['semantic_level']['B'] for plan in plans],
        dtype=torch.float32,
        device=device)
    level_c = torch.tensor(
        [plan['semantic_level']['C'] for plan in plans],
        dtype=torch.float32,
        device=device)
    level_d = torch.tensor(
        [plan['semantic_level']['D'] for plan in plans],
        dtype=torch.float32,
        device=device)

    required_snr_db = self._required_snr_db_tensor(
        level_a, level_b, level_c, level_d, semantic_req)
    required_snr_linear = torch.pow(10.0, required_snr_db / 10.0)
    denominator = torch.clamp(noise_power + interference, min=1e-18)
    channel_coeff = channel_gain / denominator
    required_power = required_snr_linear / torch.clamp(channel_coeff, min=1e-18)
    lower = torch.clamp(required_power, min=power_min)
    upper = torch.full_like(lower, float(power_max))
    feasible = lower <= upper

    alpha = user_weight * compressed / max(delay_norm, 1e-9)
    beta = user_weight * compressed * energy_weight / max(energy_norm, 1e-9)
    chosen = torch.where(feasible, lower, upper)
    ln2 = math.log(2.0)

    def rate_fn(power):
      if power.ndim == 1:
        return bandwidths * torch.log2(1.0 + channel_coeff * power)
      return bandwidths.unsqueeze(1) * torch.log2(
          1.0 + channel_coeff.unsqueeze(1) * power)

    def numerator(power):
      if power.ndim == 1:
        return alpha + beta * power
      return alpha.unsqueeze(1) + beta.unsqueeze(1) * power

    q_value = numerator(chosen) / torch.clamp(rate_fn(chosen), min=1e-9)
    max_iters = int(self._config.get('dinkelbach_max_iters', 40))
    tolerance = float(self._config.get('dinkelbach_tolerance', 1e-6))
    for _ in range(max_iters):
      root = (q_value * bandwidths / torch.clamp(beta * ln2, min=1e-12)) - (
          1.0 / torch.clamp(channel_coeff, min=1e-18))
      candidate = torch.clamp(root, min=lower, max=upper)
      eval_points = torch.stack([lower, upper, candidate], dim=1)
      residual = numerator(eval_points) - q_value.unsqueeze(1) * rate_fn(eval_points)
      best_idx = torch.argmin(residual, dim=1)
      chosen = eval_points.gather(1, best_idx.unsqueeze(1)).squeeze(1)
      best_residual = residual.gather(1, best_idx.unsqueeze(1)).squeeze(1)
      q_value = torch.where(
          torch.abs(best_residual) <= tolerance,
          q_value,
          numerator(chosen) / torch.clamp(rate_fn(chosen), min=1e-9))

    chosen = torch.where(feasible, chosen, upper)
    snr_linear = torch.clamp(channel_coeff * chosen, min=1e-12)
    snr_db = 10.0 * torch.log10(snr_linear)
    accuracy = self._semantic_accuracy_tensor(level_a, level_b, level_c, level_d, snr_db)
    return (
        chosen.detach().cpu().tolist(),
        snr_db.detach().cpu().tolist(),
        accuracy.detach().cpu().tolist(),
        feasible.detach().cpu().tolist(),
    )

  def _solve_d2e_power_closed_form_batch(self, plans, compressed, bandwidths):
    # Final-solution form of the D2E sub-problem:
    # 1) semantic constraint -> minimum feasible transmit power
    # 2) Dinkelbach inner problem -> explicit stationary candidate + boundary projection
    return self._optimize_d2e_power_dinkelbach_batch(plans, compressed, bandwidths)

  def _weighted_objective(self, delay, energy, task):
    weights = self._effective_weights(task)
    return task['user_weight'] * ((delay / self._config['delay_normalizer']) + weights['energy'] * (energy / self._config['energy_normalizer']))

  def _available_edge_capacity(self, edge_index):
    return float(self._edge_cpu_totals[edge_index])

  def _available_e2e_rate(self, affiliated, target_edge):
    if affiliated == target_edge:
      return 0.0
    return float(self._config['e2e_capacity'])

  def _available_e2e_rate_torch(self, affiliated, target_edge):
    if affiliated == target_edge:
      return 0.0
    return float(self._config['e2e_capacity'])

  def _allocate_inverse_kkt(self, components, total_capacity):
    if not components:
      return []
    lowers = np.array([c['lower'] for c in components], dtype=np.float64)
    uppers = np.array([c['upper'] for c in components], dtype=np.float64)
    coeffs = np.array([max(c['a'], 1e-12) for c in components], dtype=np.float64)
    total_capacity = max(float(total_capacity), 1e-9)
    if lowers.sum() >= total_capacity:
      return (lowers / max(lowers.sum(), 1e-9) * total_capacity).tolist()
    if uppers.sum() <= total_capacity:
      return uppers.tolist()

    def allocate_for_lambda(lam):
      values = np.sqrt(coeffs / max(lam, 1e-18))
      return np.clip(values, lowers, uppers)

    lam_low = 1e-12
    lam_high = 1.0
    while allocate_for_lambda(lam_high).sum() > total_capacity:
      lam_high *= 2.0
    for _ in range(60):
      lam_mid = 0.5 * (lam_low + lam_high)
      alloc = allocate_for_lambda(lam_mid)
      if alloc.sum() > total_capacity:
        lam_low = lam_mid
      else:
        lam_high = lam_mid
    return allocate_for_lambda(lam_high).tolist()

  def _allocate_inverse_kkt_torch(self, components, total_capacity):
    if not components:
      return []
    device = self._torch_device
    lowers = torch.tensor([c['lower'] for c in components], dtype=torch.float32, device=device)
    uppers = torch.tensor([c['upper'] for c in components], dtype=torch.float32, device=device)
    coeffs = torch.tensor(
        [max(c['a'], 1e-12) for c in components], dtype=torch.float32, device=device)
    total_capacity = max(float(total_capacity), 1e-9)
    if float(lowers.sum().item()) >= total_capacity:
      alloc = lowers / torch.clamp(lowers.sum(), min=1e-9) * total_capacity
      return alloc.detach().cpu().tolist()
    if float(uppers.sum().item()) <= total_capacity:
      return uppers.detach().cpu().tolist()

    def allocate_for_lambda(lam):
      lam_tensor = torch.tensor(lam, dtype=torch.float32, device=device)
      values = torch.sqrt(coeffs / torch.clamp(lam_tensor, min=1e-18))
      return torch.clamp(values, min=lowers, max=uppers)

    lam_low = 1e-12
    lam_high = 1.0
    while float(allocate_for_lambda(lam_high).sum().item()) > total_capacity:
      lam_high *= 2.0
    for _ in range(60):
      lam_mid = 0.5 * (lam_low + lam_high)
      alloc = allocate_for_lambda(lam_mid)
      if float(alloc.sum().item()) > total_capacity:
        lam_low = lam_mid
      else:
        lam_high = lam_mid
    return allocate_for_lambda(lam_high).detach().cpu().tolist()

  def _solve_e2e_rate_closed_form(self, components, total_capacity):
    # Final-solution form:
    # r_i^* = clip(sqrt(a_i / lambda), r_i^min, r_i^max),
    # where lambda is the common KKT multiplier determined by the sum-rate constraint.
    return self._allocate_inverse_kkt_torch(components, total_capacity)

  def _solve_d2e_bandwidth_closed_form(self, plans, compressed, powers, total_bandwidth):
    components = []
    noise_power = float(self._config['noise_power'])
    for idx, plan in enumerate(plans):
      task = plan['task']
      snr_linear = (task['channel_gain_d2e'] * float(powers[idx].item())) / max(
          noise_power + task['interference_d2e_power'], 1e-18)
      spectral_eff = max(math.log2(1.0 + max(snr_linear, 1e-12)), 1e-9)
      coeff = (
          task['user_weight'] * float(compressed[idx].item()) *
          (1.0 / max(self._config['delay_normalizer'], 1e-9) +
           task['energy_weight'] * float(powers[idx].item()) / max(self._config['energy_normalizer'], 1e-9))
          / spectral_eff)
      components.append({
          'a': coeff,
          'lower': 1e-6,
          'upper': float(total_bandwidth),
      })
    allocations = self._allocate_inverse_kkt_torch(components, total_bandwidth)
    return torch.tensor(allocations, dtype=torch.float32, device=self._torch_device)

  def _allocate_cpu_kkt(self, components, total_capacity):
    if not components:
      return []
    lowers = np.array([c['lower'] for c in components], dtype=np.float64)
    uppers = np.array([c['upper'] for c in components], dtype=np.float64)
    total_capacity = max(float(total_capacity), 1e-9)
    if lowers.sum() >= total_capacity:
      return (lowers / max(lowers.sum(), 1e-9) * total_capacity).tolist()

    unconstrained = np.array([
        self._solve_cpu_stationary(c['a'], c['b'], 0.0, c['lower'], c['upper'])
        for c in components
    ], dtype=np.float64)
    if unconstrained.sum() <= total_capacity:
      return unconstrained.tolist()

    def allocate_for_lambda(lam):
      return np.array([
          self._solve_cpu_stationary(c['a'], c['b'], lam, c['lower'], c['upper'])
          for c in components
      ], dtype=np.float64)

    lam_low = 0.0
    lam_high = 1.0
    while allocate_for_lambda(lam_high).sum() > total_capacity:
      lam_high *= 2.0
    for _ in range(60):
      lam_mid = 0.5 * (lam_low + lam_high)
      alloc = allocate_for_lambda(lam_mid)
      if alloc.sum() > total_capacity:
        lam_low = lam_mid
      else:
        lam_high = lam_mid
    return allocate_for_lambda(lam_high).tolist()

  def _allocate_cpu_kkt_torch(self, components, total_capacity):
    if not components:
      return []
    device = self._torch_device
    lowers = torch.tensor([c['lower'] for c in components], dtype=torch.float32, device=device)
    uppers = torch.tensor([c['upper'] for c in components], dtype=torch.float32, device=device)
    coeff_delay = torch.tensor(
        [max(c['a'], 1e-12) for c in components], dtype=torch.float32, device=device)
    coeff_energy = torch.tensor(
        [max(c['b'], 1e-12) for c in components], dtype=torch.float32, device=device)
    total_capacity = max(float(total_capacity), 1e-9)
    if float(lowers.sum().item()) >= total_capacity:
      alloc = lowers / torch.clamp(lowers.sum(), min=1e-9) * total_capacity
      return alloc.detach().cpu().tolist()

    unconstrained = self._solve_cpu_stationary_closed_form_torch(
        coeff_delay, coeff_energy, 0.0, lowers, uppers)
    if float(unconstrained.sum().item()) <= total_capacity:
      return unconstrained.detach().cpu().tolist()

    def allocate_for_lambda(lam):
      return self._solve_cpu_stationary_closed_form_torch(
          coeff_delay, coeff_energy, lam, lowers, uppers)

    lam_low = 0.0
    lam_high = 1.0
    while float(allocate_for_lambda(lam_high).sum().item()) > total_capacity:
      lam_high *= 2.0
    for _ in range(60):
      lam_mid = 0.5 * (lam_low + lam_high)
      alloc = allocate_for_lambda(lam_mid)
      if float(alloc.sum().item()) > total_capacity:
        lam_low = lam_mid
      else:
        lam_high = lam_mid
    return allocate_for_lambda(lam_high).detach().cpu().tolist()

  def _solve_edge_cpu_closed_form(self, components, total_capacity):
    # Final-solution form:
    # f_i^* solves 2 b_i f_i^3 + lambda f_i^2 - a_i = 0 with box constraints,
    # where lambda is the common KKT multiplier from the total edge CPU constraint.
    return self._allocate_cpu_kkt_torch(components, total_capacity)

  def _solve_cpu_stationary(self, coeff_delay, coeff_energy, lam, lower, upper):
    coeff_delay = max(float(coeff_delay), 1e-12)
    coeff_energy = max(float(coeff_energy), 1e-12)
    lower = float(lower)
    upper = float(upper)

    def derivative(value):
      return (-coeff_delay / max(value ** 2, 1e-18)) + (2.0 * coeff_energy * value) + lam

    if derivative(lower) >= 0.0:
      return lower
    if derivative(upper) <= 0.0:
      return upper
    left, right = lower, upper
    for _ in range(60):
      mid = 0.5 * (left + right)
      if derivative(mid) <= 0.0:
        left = mid
      else:
        right = mid
    return 0.5 * (left + right)

  def _solve_cpu_stationary_closed_form_torch(self, coeff_delay, coeff_energy, lam, lower, upper):
    lam_tensor = torch.as_tensor(lam, dtype=torch.float32, device=self._torch_device)
    a = torch.clamp(coeff_delay, min=1e-12)
    b = torch.clamp(coeff_energy, min=1e-12)
    c = lam_tensor / (2.0 * b)
    p = -(c ** 2) / 3.0
    q = (2.0 * (c ** 3)) / 27.0 - a / (2.0 * b)
    delta = torch.clamp((q / 2.0) ** 2 + (p / 3.0) ** 3, min=0.0)
    sqrt_delta = torch.sqrt(delta)
    u = torch.sign(-q / 2.0 + sqrt_delta) * torch.abs(-q / 2.0 + sqrt_delta).pow(1.0 / 3.0)
    v = torch.sign(-q / 2.0 - sqrt_delta) * torch.abs(-q / 2.0 - sqrt_delta).pow(1.0 / 3.0)
    stationary = u + v - c / 3.0
    stationary = torch.clamp(stationary, min=lower, max=upper)

    deriv_lower = (-a / torch.clamp(lower ** 2, min=1e-18)) + (2.0 * b * lower) + lam_tensor
    deriv_upper = (-a / torch.clamp(upper ** 2, min=1e-18)) + (2.0 * b * upper) + lam_tensor
    stationary = torch.where(deriv_lower >= 0.0, lower, stationary)
    stationary = torch.where(deriv_upper <= 0.0, upper, stationary)
    return stationary

  def _d2e_snr(self, task, power):
    numerator = power * task['channel_gain_d2e']
    denominator = self._config['noise_power'] + task['interference_d2e_power']
    return max(numerator / max(denominator, 1e-18), 1e-12)

  def _compressed_bits(self, task, level):
    b_eq = float(self._config.get('complex_symbol_bit_equivalent', 1.0))
    return float(task['sentence_len'] * level['eta'] * b_eq)

  def _effective_weights(self, task):
    return {
        'energy': task['energy_weight'],
        'penalty': float(self._config['system_weights']['penalty']),
    }

  def _cpu_delay_coeff(self, plan, kind):
    task = plan['task']
    if kind == 'decode':
      payload = plan['compressed_bits']
    else:
      payload = task['workload']
    return task['user_weight'] * payload / max(self._config['delay_normalizer'], 1e-9)

  def _cpu_energy_coeff(self, plan, kind):
    task = plan['task']
    if kind == 'decode':
      payload = plan['compressed_bits']
      alpha = self._config['alpha_decode']
    else:
      payload = task['workload']
      alpha = self._config['alpha_edge']
    return (
        task['user_weight'] * task['energy_weight'] * alpha * payload /
        max(self._config['energy_normalizer'], 1e-9))

  def _e2e_delay_energy_coeff(self, plan):
    task = plan['task']
    payload = plan['compressed_bits']
    return task['user_weight'] * payload * (
        1.0 / max(self._config['delay_normalizer'], 1e-9) +
        task['energy_weight'] * self._config['edge_forward_power'] /
        max(self._config['energy_normalizer'], 1e-9))

  def _minimum_power_for_semantic(self, task, level):
    required_snr_db = self._required_snr_db(level)
    required_snr_linear = 10.0 ** (required_snr_db / 10.0)
    denominator = self._config['noise_power'] + task['interference_d2e_power']
    return required_snr_linear * denominator / max(task['channel_gain_d2e'], 1e-18)

  def _required_snr_db(self, level):
    target = float(self._config['semantic_requirement'])
    lower = max(target - float(level['D']), 1e-9)
    upper = max(float(level['A']) + float(level['D']) - target, 1e-9)
    ratio = lower / upper
    return float(level['C'] - math.log(ratio) / max(float(level['B']), 1e-9))

  def _required_snr_db_tensor(self, level_a, level_b, level_c, level_d, target):
    lower = torch.clamp(target - level_d, min=1e-9)
    upper = torch.clamp(level_a + level_d - target, min=1e-9)
    ratio = lower / upper
    return level_c - torch.log(ratio) / torch.clamp(level_b, min=1e-9)

  def _semantic_accuracy(self, level, snr_db):
    acc = level['A'] / (1.0 + np.exp(-level['B'] * (snr_db - level['C']))) + level['D']
    return float(np.clip(acc, 0.0, 1.0))

  def _semantic_accuracy_tensor(self, level_a, level_b, level_c, level_d, snr_db):
    acc = level_a / (1.0 + torch.exp(-level_b * (snr_db - level_c))) + level_d
    return torch.clamp(acc, 0.0, 1.0)

  def _linear_to_db(self, value):
    return 10.0 * math.log10(max(value, 1e-12))

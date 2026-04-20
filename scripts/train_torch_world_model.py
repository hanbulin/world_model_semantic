import argparse
import json
import pathlib
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from envs.semantic_gym_env import SemanticGymEnv


class ReplayBuffer:

  def __init__(self, capacity, obs_dim, action_dims):
    self.capacity = capacity
    self.obs = np.zeros((capacity, obs_dim), np.float32)
    self.next_obs = np.zeros((capacity, obs_dim), np.float32)
    self.actions = np.zeros((capacity, len(action_dims)), np.int64)
    self.rewards = np.zeros((capacity, 1), np.float32)
    self.dones = np.zeros((capacity, 1), np.float32)
    self.idx = 0
    self.full = False

  def add(self, obs, action, reward, next_obs, done):
    self.obs[self.idx] = obs
    self.actions[self.idx] = action
    self.rewards[self.idx] = reward
    self.next_obs[self.idx] = next_obs
    self.dones[self.idx] = done
    self.idx = (self.idx + 1) % self.capacity
    self.full = self.full or self.idx == 0

  def __len__(self):
    return self.capacity if self.full else self.idx

  def sample(self, batch_size, device):
    idxs = np.random.randint(0, len(self), size=batch_size)
    return {
        'obs': torch.as_tensor(self.obs[idxs], device=device),
        'actions': torch.as_tensor(self.actions[idxs], device=device),
        'rewards': torch.as_tensor(self.rewards[idxs], device=device),
        'next_obs': torch.as_tensor(self.next_obs[idxs], device=device),
        'dones': torch.as_tensor(self.dones[idxs], device=device),
    }


class Encoder(nn.Module):

  def __init__(self, obs_dim, hidden_dim, latent_dim):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, latent_dim))

  def forward(self, obs):
    return self.net(obs)


class MultiHeadActor(nn.Module):

  def __init__(self, latent_dim, hidden_dim, action_dims):
    super().__init__()
    self.action_dims = action_dims
    self.group_specs = self._build_group_specs(action_dims)
    self.backbone = nn.Sequential(
        nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
    self.group_heads = nn.ModuleDict({
        name: nn.Linear(hidden_dim, size * len(indices))
        for name, size, indices in self.group_specs
    })

  @staticmethod
  def _build_group_specs(action_dims):
    ordered = []
    groups = {}
    for idx, size in enumerate(action_dims):
      key = str(size)
      if key not in groups:
        groups[key] = []
        ordered.append((key, size, groups[key]))
      groups[key].append(idx)
    return ordered

  def forward(self, latent):
    feat = self.backbone(latent)
    logits = [None] * len(self.action_dims)
    for name, size, indices in self.group_specs:
      group_logits = self.group_heads[name](feat)
      group_logits = group_logits.view(latent.shape[0], len(indices), size)
      for offset, action_idx in enumerate(indices):
        logits[action_idx] = group_logits[:, offset, :]
    return logits

  def sample(self, latent):
    feat = self.backbone(latent)
    action_slots = [None] * len(self.action_dims)
    log_probs, entropies = [], []
    for name, size, indices in self.group_specs:
      group_logits = self.group_heads[name](feat)
      group_logits = group_logits.view(latent.shape[0], len(indices), size)
      dist = torch.distributions.Categorical(logits=group_logits)
      group_actions = dist.sample()
      for offset, action_idx in enumerate(indices):
        action_slots[action_idx] = group_actions[:, offset:offset + 1]
      log_probs.append(dist.log_prob(group_actions))
      entropies.append(dist.entropy())
    actions = torch.cat(action_slots, dim=-1)
    return (
        actions,
        torch.cat(log_probs, dim=-1).sum(dim=-1, keepdim=True),
        torch.cat(entropies, dim=-1).sum(dim=-1, keepdim=True),
    )


class Critic(nn.Module):

  def __init__(self, latent_dim, hidden_dim):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, 1))

  def forward(self, latent):
    return self.net(latent)


class WorldModel(nn.Module):

  def __init__(self, obs_dim, hidden_dim, latent_dim, action_dims):
    super().__init__()
    self.encoder = Encoder(obs_dim, hidden_dim, latent_dim)
    self.action_dims = action_dims
    self.group_specs = MultiHeadActor._build_group_specs(action_dims)
    action_total = sum(action_dims)
    self.transition = nn.Sequential(
        nn.Linear(latent_dim + action_total, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, latent_dim))
    self.decoder = nn.Sequential(
        nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, obs_dim))
    self.reward_head = nn.Sequential(
        nn.Linear(latent_dim + action_total, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, 1))
    self.done_head = nn.Sequential(
        nn.Linear(latent_dim + action_total, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, 1))

  def encode_action(self, actions):
    onehots = []
    for _, size, indices in self.group_specs:
      idx_tensor = torch.as_tensor(indices, device=actions.device)
      group_actions = actions.index_select(1, idx_tensor)
      group_onehot = F.one_hot(group_actions, size).float()
      onehots.append(group_onehot.reshape(actions.shape[0], -1))
    return torch.cat(onehots, dim=-1)

  def imagine(self, latent, actions):
    action_embed = self.encode_action(actions)
    inp = torch.cat([latent, action_embed], dim=-1)
    next_latent = self.transition(inp)
    reward = self.reward_head(inp)
    done_logit = self.done_head(inp)
    return next_latent, reward, done_logit


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


def choose_device(name):
  if name == 'auto':
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  return torch.device(name)


def create_grad_scaler(use_amp, device):
  enabled = use_amp and device.type == 'cuda'
  if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
    return torch.amp.GradScaler('cuda', enabled=enabled)
  if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'GradScaler'):
    return torch.cuda.amp.GradScaler(enabled=enabled)
  return None


def autocast_context(use_amp, device):
  enabled = use_amp and device.type == 'cuda'
  if not enabled:
    return nullcontext()
  if hasattr(torch, 'autocast'):
    return torch.autocast(device_type=device.type, dtype=torch.float16)
  if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
    return torch.cuda.amp.autocast()
  return nullcontext()


def scaler_scale_loss(scaler, loss):
  return scaler.scale(loss) if scaler is not None else loss


def scaler_step(scaler, optimizer):
  if scaler is not None:
    scaler.step(optimizer)
  else:
    optimizer.step()


def scaler_update(scaler):
  if scaler is not None:
    scaler.update()


def scaler_unscale(scaler, optimizer):
  if scaler is not None:
    scaler.unscale_(optimizer)


def maybe_compile(module, enabled):
  if enabled and hasattr(torch, 'compile'):
    return torch.compile(module)
  return module


def evaluate_policy(env, actor, world_model, device, episodes):
  scores = []
  for _ in range(episodes):
    obs = env.reset()
    done = False
    total = 0.0
    while not done:
      obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
      with torch.no_grad():
        latent = world_model.encoder(obs_t)
        actions, _, _ = actor.sample(latent)
      obs, reward, done, _ = env.step(actions.squeeze(0).cpu().numpy())
      total += reward
    scores.append(total)
  return float(np.mean(scores))


def smooth(values, weight=0.9):
  out = []
  last = None
  for value in values:
    last = value if last is None else weight * last + (1 - weight) * value
    out.append(last)
  return out


def plot_rewards(logdir, rewards, eval_steps, eval_rewards):
  plt.figure(figsize=(8, 5))
  episodes = np.arange(len(rewards), dtype=np.int32)
  plt.plot(episodes, smooth(rewards), label='Smoothed reward', linewidth=2)
  plt.xlim(left=0)
  plt.xlabel('Episode index')
  plt.ylabel('Reward')
  plt.title('PyTorch world model reward convergence')
  plt.grid(True, alpha=0.3)
  plt.legend()
  out = logdir / 'reward_curve.png'
  plt.tight_layout()
  plt.savefig(out, dpi=200)
  plt.close()
  return out


def plot_episode_metric(logdir, values, ylabel, title, filename):
  plt.figure(figsize=(8, 5))
  episodes = np.arange(len(values), dtype=np.int32)
  plt.plot(episodes, values, linewidth=2)
  plt.xlim(left=0)
  plt.xlabel('Episode index')
  plt.ylabel(ylabel)
  plt.title(title)
  plt.grid(True, alpha=0.3)
  out = logdir / filename
  plt.tight_layout()
  plt.savefig(out, dpi=200)
  plt.close()
  return out


def flush_training_artifacts(logdir, episode_rewards, metrics_log, eval_steps, eval_rewards, device):
  (logdir / 'episode_rewards.json').write_text(
      json.dumps(episode_rewards, indent=2), encoding='utf-8')
  (logdir / 'metrics.json').write_text(
      json.dumps({
          'episodes': metrics_log,
          'eval_steps': eval_steps,
          'eval_rewards': eval_rewards,
          'device': str(device),
      }, indent=2), encoding='utf-8')


def format_seconds(seconds):
  seconds = max(int(seconds), 0)
  hours = seconds // 3600
  minutes = (seconds % 3600) // 60
  secs = seconds % 60
  return f'{hours:02d}:{minutes:02d}:{secs:02d}'


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', default='configs/pytorch_world_model.json')
  args = parser.parse_args()

  root = pathlib.Path(__file__).resolve().parents[1]
  cfg_path = root / args.config if not pathlib.Path(args.config).is_absolute() else pathlib.Path(args.config)
  config = json.loads(cfg_path.read_text(encoding='utf-8'))
  set_seed(config['seed'])
  device = choose_device(config['device'])
  torch.set_float32_matmul_precision(config.get('matmul_precision', 'high'))
  use_amp = bool(config.get('use_amp', device.type == 'cuda'))
  compile_models = bool(config.get('compile_models', device.type == 'cuda'))
  updates_per_step = int(config.get('updates_per_step', 1))

  logdir = root / config['logdir']
  logdir.mkdir(parents=True, exist_ok=True)
  (logdir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

  env = SemanticGymEnv(config['env_config'], seed=config['seed'])
  eval_env = SemanticGymEnv(config['env_config'], seed=config['seed'] + 1000)
  world_model = WorldModel(
      env.obs_dim, config['hidden_dim'], config['latent_dim'], env.action_dims).to(device)
  actor = MultiHeadActor(config['latent_dim'], config['hidden_dim'], env.action_dims).to(device)
  critic = Critic(config['latent_dim'], config['hidden_dim']).to(device)
  target_critic = Critic(config['latent_dim'], config['hidden_dim']).to(device)
  world_model = maybe_compile(world_model, compile_models)
  actor = maybe_compile(actor, compile_models)
  critic = maybe_compile(critic, compile_models)
  target_critic = maybe_compile(target_critic, compile_models)
  target_critic.load_state_dict(critic.state_dict())

  wm_opt = torch.optim.Adam(world_model.parameters(), lr=config['world_model_lr'])
  actor_opt = torch.optim.Adam(actor.parameters(), lr=config['actor_lr'])
  critic_opt = torch.optim.Adam(critic.parameters(), lr=config['critic_lr'])
  scaler = create_grad_scaler(use_amp, device)

  start_step = 0
  resume_path = config.get('resume_checkpoint')
  if resume_path:
    ckpt_path = root / resume_path if not pathlib.Path(resume_path).is_absolute() else pathlib.Path(resume_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    world_model.load_state_dict(checkpoint['world_model'])
    actor.load_state_dict(checkpoint['actor'])
    critic.load_state_dict(checkpoint['critic'])
    if 'target_critic' in checkpoint:
      target_critic.load_state_dict(checkpoint['target_critic'])
    else:
      target_critic.load_state_dict(checkpoint['critic'])
    if 'wm_opt' in checkpoint:
      wm_opt.load_state_dict(checkpoint['wm_opt'])
    if 'actor_opt' in checkpoint:
      actor_opt.load_state_dict(checkpoint['actor_opt'])
    if 'critic_opt' in checkpoint:
      critic_opt.load_state_dict(checkpoint['critic_opt'])
    if 'scaler' in checkpoint and checkpoint['scaler'] is not None:
      scaler.load_state_dict(checkpoint['scaler'])
    start_step = int(checkpoint.get('step', 0))
    print(f'resumed from {ckpt_path} at step={start_step}', flush=True)

  replay = ReplayBuffer(config['replay_size'], env.obs_dim, env.action_dims)

  obs = env.reset()
  episode_reward = 0.0
  episode_delay_sum = 0.0
  episode_energy_sum = 0.0
  episode_step_count = 0
  episode_rewards = []
  eval_steps, eval_rewards = [], []
  metrics_log = []
  train_log_path = logdir / 'training_log.jsonl'
  if train_log_path.exists():
    train_log_path.unlink()
  train_start_time = time.time()

  if start_step >= config['total_steps']:
    raise ValueError(
        f"resume step {start_step} is not smaller than total_steps {config['total_steps']}")

  for step in range(start_step + 1, config['total_steps'] + 1):
    if step <= config['seed_steps']:
      action = env.sample_random_action()
    else:
      obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
      with torch.no_grad():
        latent = world_model.encoder(obs_t)
        action, _, _ = actor.sample(latent)
      action = action.squeeze(0).cpu().numpy()

    next_obs, reward, done, info = env.step(action)
    replay.add(obs, action, reward * config['reward_scale'], next_obs, float(done))
    obs = next_obs
    episode_reward += reward
    episode_delay_sum += float(info.get('total_delay', 0.0))
    episode_energy_sum += float(info.get('total_energy', 0.0))
    episode_step_count += 1

    if done:
      avg_total_delay = episode_delay_sum / max(episode_step_count, 1)
      avg_total_energy = episode_energy_sum / max(episode_step_count, 1)
      episode_rewards.append(episode_reward)
      metrics_log.append({
          'episode': len(episode_rewards),
          'reward': episode_reward,
          'avg_total_delay': avg_total_delay,
          'avg_total_energy': avg_total_energy,
      })
      episode_entry = {
          'kind': 'episode',
          'episode': len(episode_rewards),
          'step': step,
          'total_steps': int(config['total_steps']),
          'reward': float(episode_reward),
          'avg_total_delay': float(avg_total_delay),
          'avg_total_energy': float(avg_total_energy),
          'elapsed_seconds': time.time() - train_start_time,
      }
      with train_log_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(episode_entry, ensure_ascii=False) + '\n')
      flush_training_artifacts(logdir, episode_rewards, metrics_log, eval_steps, eval_rewards, device)
      obs = env.reset()
      episode_reward = 0.0
      episode_delay_sum = 0.0
      episode_energy_sum = 0.0
      episode_step_count = 0

    if len(replay) >= config['batch_size']:
      wm_loss = critic_loss = actor_loss = None
      amp_ctx = autocast_context(use_amp, device)
      for _ in range(updates_per_step):
        batch = replay.sample(config['batch_size'], device)
        with amp_ctx:
          latent = world_model.encoder(batch['obs'])
          next_latent_target = world_model.encoder(batch['next_obs']).detach()
          pred_next_latent, pred_reward, pred_done = world_model.imagine(latent, batch['actions'])
          recon_next_obs = world_model.decoder(pred_next_latent)

          wm_loss = (
              F.mse_loss(pred_next_latent, next_latent_target) +
              F.mse_loss(recon_next_obs, batch['next_obs']) +
              F.mse_loss(pred_reward, batch['rewards']) +
              F.binary_cross_entropy_with_logits(pred_done, batch['dones']))
        wm_opt.zero_grad(set_to_none=True)
        scaler_scale_loss(scaler, wm_loss).backward()
        scaler_unscale(scaler, wm_opt)
        nn.utils.clip_grad_norm_(world_model.parameters(), config['grad_clip'])
        scaler_step(scaler, wm_opt)
        scaler_update(scaler)

        latent_detached = latent.detach()
        next_latent_detached = next_latent_target.detach()
        with torch.no_grad():
          with amp_ctx:
            target_bootstrap = target_critic(next_latent_detached)
          target_value = batch['rewards'] + config['gamma'] * (1 - batch['dones']) * target_bootstrap.float()
        with amp_ctx:
          value = critic(latent_detached)
          critic_loss = F.mse_loss(value, target_value)
        critic_opt.zero_grad(set_to_none=True)
        scaler_scale_loss(scaler, critic_loss).backward()
        scaler_unscale(scaler, critic_opt)
        nn.utils.clip_grad_norm_(critic.parameters(), config['grad_clip'])
        scaler_step(scaler, critic_opt)
        scaler_update(scaler)

        imagined_latent = latent_detached
        with amp_ctx:
          actor_loss = 0.0
          for _ in range(config['imagination_horizon']):
            act, log_prob, entropy = actor.sample(imagined_latent)
            imagined_latent, imagined_reward, imagined_done = world_model.imagine(imagined_latent, act)
            imagined_value = critic(imagined_latent)
            advantage = imagined_reward + config['gamma'] * (1 - torch.sigmoid(imagined_done)) * imagined_value
            actor_loss = actor_loss + (-(advantage.detach() * log_prob) - config['entropy_coef'] * entropy).mean()
          actor_loss = actor_loss / config['imagination_horizon']
        actor_opt.zero_grad(set_to_none=True)
        scaler_scale_loss(scaler, actor_loss).backward()
        scaler_unscale(scaler, actor_opt)
        nn.utils.clip_grad_norm_(actor.parameters(), config['grad_clip'])
        scaler_step(scaler, actor_opt)
        scaler_update(scaler)

        tau = 1 - config['lambda_value']
        for src, dst in zip(critic.parameters(), target_critic.parameters()):
          dst.data.mul_(1 - tau).add_(tau * src.data)

      if step % config['log_every'] == 0:
        elapsed = time.time() - train_start_time
        progress = step / max(config['total_steps'], 1)
        eta_seconds = elapsed * (config['total_steps'] - step) / max(step, 1)
        recent_rewards = episode_rewards[-10:]
        avg_recent_reward = float(np.mean(recent_rewards)) if recent_rewards else None
        last_reward = float(episode_rewards[-1]) if episode_rewards else None
        train_entry = {
            'kind': 'train',
            'step': step,
            'total_steps': int(config['total_steps']),
            'progress': progress,
            'elapsed_seconds': elapsed,
            'eta_seconds': eta_seconds,
            'replay': len(replay),
            'updates_per_step': updates_per_step,
            'wm_loss': float(wm_loss.item()),
            'critic_loss': float(critic_loss.item()),
            'actor_loss': float(actor_loss.item()),
            'episode_count': len(episode_rewards),
            'last_episode_reward': last_reward,
            'avg_recent_reward': avg_recent_reward,
        }
        with train_log_path.open('a', encoding='utf-8') as f:
          f.write(json.dumps(train_entry, ensure_ascii=False) + '\n')
        flush_training_artifacts(logdir, episode_rewards, metrics_log, eval_steps, eval_rewards, device)
        print(
            f"step={step}/{config['total_steps']} progress={progress:.1%} "
            f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta_seconds)} "
            f"replay={len(replay)} updates={updates_per_step} "
            f"wm_loss={wm_loss.item():.4f} critic_loss={critic_loss.item():.4f} "
            f"actor_loss={actor_loss.item():.4f}",
            flush=True)

    if step % config['eval_every'] == 0 and step > config['seed_steps']:
      eval_reward = evaluate_policy(eval_env, actor, world_model, device, config['eval_episodes'])
      eval_steps.append(len(episode_rewards))
      eval_rewards.append(eval_reward)
      elapsed = time.time() - train_start_time
      progress = step / max(config['total_steps'], 1)
      eta_seconds = elapsed * (config['total_steps'] - step) / max(step, 1)
      eval_entry = {
          'kind': 'eval',
          'step': step,
          'total_steps': int(config['total_steps']),
          'progress': progress,
          'elapsed_seconds': elapsed,
          'eta_seconds': eta_seconds,
          'avg_reward': float(eval_reward),
          'episode_count': len(episode_rewards),
      }
      with train_log_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(eval_entry, ensure_ascii=False) + '\n')
      flush_training_artifacts(logdir, episode_rewards, metrics_log, eval_steps, eval_rewards, device)
      print(
          f'eval step={step}/{config["total_steps"]} progress={progress:.1%} '
          f'elapsed={format_seconds(elapsed)} eta={format_seconds(eta_seconds)} '
          f'avg_reward={eval_reward:.3f}',
          flush=True)

    if step % config['save_every'] == 0:
      torch.save({
          'world_model': world_model.state_dict(),
          'actor': actor.state_dict(),
          'critic': critic.state_dict(),
          'target_critic': target_critic.state_dict(),
          'wm_opt': wm_opt.state_dict(),
          'actor_opt': actor_opt.state_dict(),
          'critic_opt': critic_opt.state_dict(),
          'scaler': scaler.state_dict() if scaler is not None else None,
          'step': step,
      }, logdir / f'checkpoint_{step:06d}.pt')
      flush_training_artifacts(logdir, episode_rewards, metrics_log, eval_steps, eval_rewards, device)

  flush_training_artifacts(logdir, episode_rewards, metrics_log, eval_steps, eval_rewards, device)
  reward_plot = plot_rewards(logdir, episode_rewards, eval_steps, eval_rewards)
  delay_plot = plot_episode_metric(
      logdir,
      [entry['avg_total_delay'] for entry in metrics_log],
      'Average total time',
      'Average total time with the increase of episodes',
      'avg_total_time_curve.png')
  energy_plot = plot_episode_metric(
      logdir,
      [entry['avg_total_energy'] for entry in metrics_log],
      'Average total energy consumption',
      'Average total energy consumption with the increase of episodes',
      'avg_total_energy_curve.png')
  print(f'saved reward plot to {reward_plot}', flush=True)
  print(f'saved average total time plot to {delay_plot}', flush=True)
  print(f'saved average total energy plot to {energy_plot}', flush=True)


if __name__ == '__main__':
  main()

"""
PPO training loop with theory-guided regularization.

Loss = L_PPO + λ_d * L_prior^d + λ_c * L_prior^c + λ_g * L_game

Usage:
    python tactical_acados/rl/train.py --scenario scenario_a --max-episodes 100
"""

import os
import sys
import time
import argparse
import numpy as np
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available. Training requires PyTorch.")

dir_path = os.path.dirname(os.path.abspath(__file__))
tactical_dir = os.path.join(dir_path, '..')
sys.path.insert(0, tactical_dir)
sys.path.insert(0, os.path.join(tactical_dir, '..', 'src'))

from config import TacticalConfig, DEFAULT_CONFIG
from tactical_action import NUM_DISCRETE_ACTIONS
from observation import TacticalObservation


class RolloutBuffer:
    """Stores rollout data for PPO updates."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.discrete_actions = []
        self.cont_actions = []
        self.p2p_actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.safe_masks = []
        self.game_value_targets = []

    def add(self, obs, discrete_action, cont_action, p2p_action,
            log_prob, reward, value, done, safe_mask, game_value_target=0.0):
        self.obs.append(obs)
        self.discrete_actions.append(discrete_action)
        self.cont_actions.append(cont_action)
        self.p2p_actions.append(p2p_action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.safe_masks.append(safe_mask)
        self.game_value_targets.append(game_value_target)

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        """Compute GAE advantages and discounted returns."""
        n = len(self.rewards)
        advantages = np.zeros(n)
        last_gae = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_done = 0
            else:
                next_value = self.values[t + 1]
                next_done = self.dones[t + 1]

            delta = (self.rewards[t] + gamma * next_value * (1 - next_done)
                     - self.values[t])
            last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values)
        return returns, advantages

    def to_tensors(self, returns, advantages):
        """Convert to PyTorch tensors."""
        return {
            'obs': torch.FloatTensor(np.array(self.obs)),
            'discrete_actions': torch.LongTensor(np.array(self.discrete_actions)),
            'cont_actions': torch.FloatTensor(np.array(self.cont_actions)),
            'p2p_actions': torch.FloatTensor(np.array(self.p2p_actions)),
            'old_log_probs': torch.FloatTensor(np.array(self.log_probs)),
            'returns': torch.FloatTensor(returns),
            'advantages': torch.FloatTensor(advantages),
            'safe_masks': torch.FloatTensor(np.array(self.safe_masks)),
            'game_value_targets': torch.FloatTensor(np.array(self.game_value_targets)),
        }

    def __len__(self):
        return len(self.rewards)


def train(
        scenario_name: str = 'scenario_a',
        max_episodes: int = 1000,
        max_steps_per_episode: int = 200,
        save_dir: str = None,
        cfg: TacticalConfig = None,
):
    """Main PPO training loop."""
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch required for training.")
        return

    if cfg is None:
        cfg = TacticalConfig()

    if save_dir is None:
        save_dir = os.path.join(tactical_dir, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    # Import components
    from tactical_env import TacticalRacingEnv
    from hybrid_ppo import HybridPPOPolicy
    from theory_prior import TheoryPrior

    # Create environment
    env = TacticalRacingEnv(
        scenario_name=scenario_name,
        cfg=cfg,
        max_steps=max_steps_per_episode,
    )

    # Create policy
    obs_dim = TacticalObservation.obs_dim(cfg)
    policy = HybridPPOPolicy(obs_dim=obs_dim, cfg=cfg)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.ppo_lr)
    theory_prior = TheoryPrior(cfg)

    # Training stats
    stats = defaultdict(list)
    best_mean_reward = -float('inf')

    print("=" * 70)
    print(f"Training Hybrid PPO on {scenario_name}")
    print(f"  Obs dim: {obs_dim}, Discrete: {NUM_DISCRETE_ACTIONS}, Continuous: 5")
    print(f"  Episodes: {max_episodes}, Steps/ep: {max_steps_per_episode}")
    print("=" * 70)

    for episode in range(max_episodes):
        t_start = time.time()
        buffer = RolloutBuffer()

        obs, info = env.reset(seed=episode)
        safe_mask = info.get('safe_mask', np.ones(NUM_DISCRETE_ACTIONS))
        episode_reward = 0.0
        episode_length = 0

        for step in range(max_steps_per_episode):
            # Get action from policy
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            mask_tensor = torch.FloatTensor(safe_mask).unsqueeze(0)

            with torch.no_grad():
                action_out = policy.get_action_and_value(
                    obs_tensor, mask_tensor,
                )

            # Extract action values
            discrete = int(action_out['discrete_action'].item())
            continuous = action_out['cont_action'].squeeze(0).numpy()
            p2p = float(action_out['p2p_action'].item())
            log_prob = float(action_out['log_prob'].item())
            value = float(action_out['value'].item())

            # Step environment
            action_dict = {
                'discrete': discrete,
                'continuous': continuous,
                'p2p': int(p2p),
            }
            next_obs, reward, terminated, truncated, info = env.step(action_dict)
            done = terminated or truncated

            # Store
            buffer.add(
                obs=obs,
                discrete_action=discrete,
                cont_action=continuous,
                p2p_action=p2p,
                log_prob=log_prob,
                reward=reward,
                value=value,
                done=float(done),
                safe_mask=safe_mask,
            )

            episode_reward += reward
            episode_length += 1
            obs = next_obs
            safe_mask = info.get('safe_mask', np.ones(NUM_DISCRETE_ACTIONS))

            if done:
                break

        # Compute returns and advantages
        with torch.no_grad():
            last_obs = torch.FloatTensor(obs).unsqueeze(0)
            last_value = float(policy.forward(last_obs)['value'].item())

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, cfg.ppo_gamma, cfg.ppo_gae_lambda,
        )

        # PPO update
        data = buffer.to_tensors(returns, advantages)
        ppo_losses = _ppo_update(policy, optimizer, data, theory_prior, cfg)

        # Logging
        t_elapsed = time.time() - t_start
        stats['episode_reward'].append(episode_reward)
        stats['episode_length'].append(episode_length)

        if episode % 5 == 0:
            mean_reward = np.mean(stats['episode_reward'][-10:])
            print(f"[Ep {episode:4d}] reward={episode_reward:7.2f} "
                  f"avg10={mean_reward:7.2f} len={episode_length:3d} "
                  f"loss_ppo={ppo_losses.get('ppo_loss', 0):.4f} "
                  f"loss_theory={ppo_losses.get('theory_loss', 0):.4f} "
                  f"time={t_elapsed:.1f}s")

        # Save best
        mean_reward = np.mean(stats['episode_reward'][-20:]) if len(stats['episode_reward']) >= 20 else episode_reward
        if mean_reward > best_mean_reward and episode >= 20:
            best_mean_reward = mean_reward
            torch.save(policy.state_dict(),
                       os.path.join(save_dir, 'best_policy.pt'))
            print(f"  → New best: {best_mean_reward:.2f}")

        # Periodic save
        if episode % 50 == 0:
            torch.save({
                'episode': episode,
                'policy_state': policy.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'stats': dict(stats),
            }, os.path.join(save_dir, f'checkpoint_{episode}.pt'))

    # Final save
    torch.save(policy.state_dict(), os.path.join(save_dir, 'final_policy.pt'))
    print(f"\nTraining complete. Best avg reward: {best_mean_reward:.2f}")
    return stats


def _ppo_update(policy, optimizer, data, theory_prior, cfg):
    """Perform PPO update epochs on collected data."""
    n_samples = len(data['obs'])
    batch_size = min(cfg.ppo_batch_size, n_samples)
    losses_log = {}

    # Normalize advantages
    advantages = data['advantages']
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for epoch in range(cfg.ppo_n_epochs):
        indices = torch.randperm(n_samples)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = indices[start:end]

            batch_obs = data['obs'][idx]
            batch_discrete = data['discrete_actions'][idx]
            batch_cont = data['cont_actions'][idx]
            batch_p2p = data['p2p_actions'][idx]
            batch_old_lp = data['old_log_probs'][idx]
            batch_returns = data['returns'][idx]
            batch_adv = advantages[idx]
            batch_mask = data['safe_masks'][idx]
            batch_gv_target = data['game_value_targets'][idx]

            # Evaluate current policy
            eval_out = policy.evaluate_actions(
                batch_obs, batch_discrete, batch_cont, batch_p2p, batch_mask,
            )

            # PPO clipped objective
            ratio = torch.exp(eval_out['log_prob'] - batch_old_lp)
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1 - cfg.ppo_clip_eps,
                                1 + cfg.ppo_clip_eps) * batch_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = 0.5 * (eval_out['value'] - batch_returns).pow(2).mean()

            # Entropy bonus
            entropy_loss = -eval_out['entropy'].mean()

            # PPO total
            ppo_loss = (policy_loss +
                        cfg.ppo_value_coef * value_loss +
                        cfg.ppo_entropy_coef * entropy_loss)

            # Theory-guided regularization
            # Game value from auxiliary head
            action_enc = policy.encode_action(batch_discrete, batch_cont, batch_p2p)
            game_value_pred = policy.compute_game_value(
                eval_out['features'], action_enc,
            )

            theory_losses = theory_prior.compute_all_losses(
                policy_probs=eval_out['discrete_probs'],
                cont_mean=eval_out['cont_mean'],
                discrete_actions=batch_discrete,
                game_value_pred=game_value_pred,
                game_value_target=batch_gv_target,
                safe_mask=batch_mask,
            )

            # Combined loss
            total_loss = ppo_loss + theory_losses['L_theory_total']

            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.ppo_max_grad_norm)
            optimizer.step()

            losses_log = {
                'ppo_loss': float(ppo_loss.item()),
                'theory_loss': float(theory_losses['L_theory_total'].item()),
                'value_loss': float(value_loss.item()),
                'policy_loss': float(policy_loss.item()),
            }

    return losses_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train tactical RL agent')
    parser.add_argument('--scenario', type=str, default='scenario_a')
    parser.add_argument('--max-episodes', type=int, default=1000)
    parser.add_argument('--max-steps-per-episode', type=int, default=200)
    parser.add_argument('--save-dir', type=str, default=None)
    args = parser.parse_args()

    train(
        scenario_name=args.scenario,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        save_dir=args.save_dir,
    )

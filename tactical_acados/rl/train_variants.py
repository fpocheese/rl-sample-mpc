"""
Unified training script for 3 RL variants.

Variants:
  A-oursrl : Our planner + Advanced RL (BC pretrain + curriculum + shaped reward + theory prior)
  oursrl   : Our planner + Basic RL   (PPO only, base reward, no advanced tricks)
  pure-rl  : No planner guidance       (PPO + sparse reward, no theory prior, no heuristic)

Usage:
  python train_variants.py --variant A-oursrl --total-episodes 600 --save-dir checkpoints/A_oursrl
  python train_variants.py --variant oursrl   --total-episodes 600 --save-dir checkpoints/oursrl
  python train_variants.py --variant pure-rl  --total-episodes 600 --save-dir checkpoints/pure_rl
"""

import os
import sys
import time
import json
import argparse
import pickle
import numpy as np
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch required.")
    sys.exit(1)

dir_path = os.path.dirname(os.path.abspath(__file__))
tactical_dir = os.path.join(dir_path, '..')
project_root = os.path.join(tactical_dir, '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, tactical_dir)
sys.path.insert(0, dir_path)

from config import TacticalConfig, DEFAULT_CONFIG
from tactical_action import NUM_DISCRETE_ACTIONS, TacticalAction, DiscreteTactic, PreferenceVector
from observation import TacticalObservation
from tactical_env import TacticalRacingEnv
from hybrid_ppo import HybridPPOPolicy
from theory_prior import TheoryPrior
from reward import RewardComputer
from reward_shaped import ShapedRewardComputer, SparseRewardComputer
from curriculum import CurriculumManager, CurriculumConfig
from train import RolloutBuffer, _ppo_update


# ═══════════════════════════════════════════════════════════════════════
# Behavioral Cloning Pretrain (A-oursrl only)
# ═══════════════════════════════════════════════════════════════════════

def bc_pretrain(policy, demo_path, cfg, n_epochs=15, batch_size=128, lr=1e-3):
    """
    Behavioral Cloning pretraining from expert demonstrations.
    Initializes the policy network close to the expert heuristic.
    """
    print("\n" + "=" * 60)
    print("Behavioral Cloning Pretrain")
    print("=" * 60)

    if not os.path.exists(demo_path):
        print(f"WARNING: No expert demos at {demo_path}, skipping BC pretrain.")
        return

    with open(demo_path, 'rb') as f:
        demos = pickle.load(f)

    n_samples = len(demos['obs'])
    print(f"  Loaded {n_samples} expert transitions")

    obs_t = torch.FloatTensor(demos['obs'])
    discrete_t = torch.LongTensor(demos['discrete_actions'])
    cont_t = torch.FloatTensor(demos['cont_actions'])
    p2p_t = torch.FloatTensor(demos['p2p_actions'])
    mask_t = torch.FloatTensor(demos['safe_masks'])

    bc_optimizer = optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(n_epochs):
        indices = torch.randperm(n_samples)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            idx = indices[start:end]

            b_obs = obs_t[idx]
            b_discrete = discrete_t[idx]
            b_cont = cont_t[idx]
            b_p2p = p2p_t[idx]
            b_mask = mask_t[idx]

            out = policy.forward(b_obs, b_mask)

            # Cross-entropy loss for discrete actions
            ce_loss = F.cross_entropy(out['discrete_logits'], b_discrete)

            # MSE loss for continuous actions
            mse_loss = F.mse_loss(out['cont_mean'], b_cont)

            # BCE loss for P2P
            bce_loss = F.binary_cross_entropy_with_logits(
                out['p2p_logit'], b_p2p,
            )

            loss = ce_loss + mse_loss + 0.5 * bce_loss

            bc_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            bc_optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if epoch % 3 == 0 or epoch == n_epochs - 1:
            print(f"  BC Epoch {epoch:3d}: loss={avg_loss:.4f}")

    print("  BC Pretrain complete.\n")


# ═══════════════════════════════════════════════════════════════════════
# Environment with variant-specific reward
# ═══════════════════════════════════════════════════════════════════════

class VariantEnv:
    """
    Wraps TacticalRacingEnv with variant-specific reward computation.
    """

    def __init__(self, variant: str, scenario_name: str, cfg: TacticalConfig,
                 max_steps: int = 500, randomize_init: bool = True):
        self.variant = variant
        self.cfg = cfg
        self.env = TacticalRacingEnv(
            scenario_name=scenario_name, cfg=cfg,
            randomize_init=randomize_init, max_steps=max_steps,
        )

        # Reward computer per variant
        if variant == 'A-oursrl':
            self.reward_comp = ShapedRewardComputer(cfg, shaping_coeff=2.0)
            # Also load heuristic policy for shaping
            sys.path.insert(0, os.path.join(tactical_dir, 'policies'))
            from heuristic_policy import HeuristicTacticalPolicy
            self.heuristic = HeuristicTacticalPolicy(cfg=cfg)
        elif variant == 'oursrl':
            self.reward_comp = RewardComputer(cfg)  # base reward
            self.heuristic = None
        elif variant == 'pure-rl':
            self.reward_comp = SparseRewardComputer(cfg)
            self.heuristic = None
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Forward properties
        self.track_handler = self.env.track_handler
        self.global_planner = self.env.global_planner
        self.observation_space_shape = self.env.observation_space_shape

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        if self.heuristic is not None:
            # Reset heuristic FSM state
            self.heuristic.phase = "RACELINE"
            self.heuristic.phase_time = 0.0
            self.heuristic._overtake_ready_ext = False
        if hasattr(self.reward_comp, 'reset'):
            self.reward_comp.reset()
        # Track previous states for sparse reward
        self._prev_ego = dict(self.env.ego_state)
        self._prev_opponents = [opp.get_state() for opp in self.env.opponents]
        return obs, info

    def step(self, action_dict):
        """Step with variant-specific reward override."""
        # Save pre-step state for sparse reward
        pre_ego = dict(self.env.ego_state)
        pre_opponents = [opp.get_state() for opp in self.env.opponents]

        # We let the env step normally to get the state transitions
        obs, base_reward, terminated, truncated, info = self.env.step(action_dict)

        # Override reward based on variant
        if self.variant == 'A-oursrl':
            # Get heuristic action for shaping
            heuristic_action = None
            if self.heuristic is not None:
                try:
                    heuristic_action = self.heuristic.act(self.env._last_obs)
                except Exception:
                    pass

            # Reconstruct TacticalAction for shaped reward
            rl_action = TacticalAction(
                discrete_tactic=DiscreteTactic(int(action_dict['discrete'])),
                aggressiveness=float(action_dict['continuous'][0]),
                preference=PreferenceVector(
                    rho_v=float(action_dict['continuous'][1]),
                    rho_n=float(action_dict['continuous'][2]),
                    rho_s=float(action_dict['continuous'][3]),
                    rho_w=float(action_dict['continuous'][4]),
                ),
                p2p_trigger=bool(action_dict.get('p2p', 0)),
            )

            # Use shaped reward
            reward_dict = info.get('reward_components', {})
            if heuristic_action is not None:
                # Add shaping bonus
                from reward_shaped import ShapedRewardComputer
                shaped_bonus = self.reward_comp._compute_potential(
                    self.env.ego_state,
                    [opp.get_state() for opp in self.env.opponents],
                    rl_action,
                    heuristic_action,
                    self.env.track_handler,
                ) * 0.5  # damped shaping
                reward = base_reward + shaped_bonus
            else:
                reward = base_reward

        elif self.variant == 'pure-rl':
            # Sparse reward — use pre-step states for delta computation
            reward_dict = self.reward_comp.compute(
                ego_state=self.env.ego_state,
                ego_state_prev=pre_ego,
                opponents=[opp.get_state() for opp in self.env.opponents],
                opponents_prev=pre_opponents,
                action=TacticalAction(
                    discrete_tactic=DiscreteTactic(int(action_dict['discrete'])),
                    aggressiveness=float(action_dict['continuous'][0]),
                    preference=PreferenceVector(*[float(x) for x in action_dict['continuous'][1:]]),
                    p2p_trigger=bool(action_dict.get('p2p', 0)),
                ),
                prev_action=self.env.prev_action,
                planner_healthy=self.env.planner.planner_healthy,
                track_handler=self.env.track_handler,
                p2p_active=self.env.p2p.active,
            )
            reward = float(reward_dict['total'])
        else:
            # oursrl: use base reward from env
            reward = base_reward

        return obs, reward, terminated, truncated, info


# ═══════════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train_variant(
        variant: str,
        total_episodes: int = 600,
        max_steps: int = 500,
        save_dir: str = None,
        scenario_name: str = 'scenario_a',
        demo_path: str = None,
        seed: int = 42,
):
    """
    Train one RL variant.

    Args:
        variant: 'A-oursrl', 'oursrl', or 'pure-rl'
        total_episodes: total training episodes
        max_steps: max steps per episode
        save_dir: checkpoint directory
        scenario_name: starting scenario
        demo_path: path to expert demos (A-oursrl only)
        seed: random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = TacticalConfig()

    if save_dir is None:
        save_dir = os.path.join(tactical_dir, 'checkpoints', variant.replace('-', '_'))
    os.makedirs(save_dir, exist_ok=True)

    # Create components
    obs_dim = TacticalObservation.obs_dim(cfg)
    policy = HybridPPOPolicy(obs_dim=obs_dim, cfg=cfg)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.ppo_lr)

    # Variant-specific setup
    use_theory_prior = (variant in ['A-oursrl', 'oursrl'])
    use_curriculum = (variant == 'A-oursrl')
    use_bc_pretrain = (variant == 'A-oursrl')

    theory_prior = TheoryPrior(cfg) if use_theory_prior else None
    curriculum = CurriculumManager() if use_curriculum else None

    # BC pretrain for A-oursrl
    if use_bc_pretrain:
        if demo_path is None:
            demo_path = os.path.join(dir_path, 'expert_demos.pkl')
        bc_pretrain(policy, demo_path, cfg, n_epochs=15, batch_size=128)

    # Create env
    env_scenario = scenario_name
    if curriculum:
        env_kwargs = curriculum.get_env_kwargs()
        env_scenario = env_kwargs['scenario_name']

    env = VariantEnv(
        variant=variant,
        scenario_name=env_scenario,
        cfg=cfg,
        max_steps=max_steps,
        randomize_init=True,
    )

    # Training stats
    stats = defaultdict(list)
    best_mean_reward = -float('inf')

    print("\n" + "=" * 70)
    print(f"Training variant: {variant}")
    print(f"  Obs dim: {obs_dim}, Actions: {NUM_DISCRETE_ACTIONS}d + 5c + 1p2p")
    print(f"  Episodes: {total_episodes}, Max steps: {max_steps}")
    print(f"  Theory prior: {use_theory_prior}")
    print(f"  Curriculum: {use_curriculum}")
    print(f"  BC pretrain: {use_bc_pretrain}")
    if curriculum:
        print(f"  Curriculum:\n{curriculum.summary()}")
    print("=" * 70 + "\n")

    global_episode = 0

    for episode in range(total_episodes):
        t_start = time.time()

        # Curriculum: maybe switch scenario
        if curriculum and not curriculum.is_complete:
            # Re-create env if scenario changed
            phase = curriculum.current_phase
            if phase.scenario_name != env_scenario:
                env_scenario = phase.scenario_name
                env = VariantEnv(
                    variant=variant,
                    scenario_name=env_scenario,
                    cfg=cfg,
                    max_steps=phase.max_steps,
                    randomize_init=phase.randomize_init,
                )

        buffer = RolloutBuffer()
        obs, info = env.reset(seed=episode + seed)
        safe_mask = info.get('safe_mask', np.ones(NUM_DISCRETE_ACTIONS))
        episode_reward = 0.0
        episode_length = 0

        for step in range(max_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            mask_tensor = torch.FloatTensor(safe_mask).unsqueeze(0)

            with torch.no_grad():
                action_out = policy.get_action_and_value(obs_tensor, mask_tensor)

            discrete = int(action_out['discrete_action'].item())
            continuous = action_out['cont_action'].squeeze(0).numpy()
            p2p = float(action_out['p2p_action'].item())
            log_prob = float(action_out['log_prob'].item())
            value = float(action_out['value'].item())

            action_dict = {
                'discrete': discrete,
                'continuous': continuous,
                'p2p': int(p2p),
            }

            next_obs, reward, terminated, truncated, info = env.step(action_dict)
            done = terminated or truncated

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

        # Compute returns
        with torch.no_grad():
            last_obs = torch.FloatTensor(obs).unsqueeze(0)
            last_value = float(policy.forward(last_obs)['value'].item())

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, cfg.ppo_gamma, cfg.ppo_gae_lambda,
        )

        # PPO update (with or without theory prior)
        data = buffer.to_tensors(returns, advantages)
        if use_theory_prior:
            ppo_losses = _ppo_update(policy, optimizer, data, theory_prior, cfg)
        else:
            ppo_losses = _ppo_update_no_theory(policy, optimizer, data, cfg)

        # Curriculum
        if curriculum:
            curriculum.record_episode(episode_reward)

        # Stats
        t_elapsed = time.time() - t_start
        stats['episode_reward'].append(float(episode_reward))
        stats['episode_length'].append(int(episode_length))
        stats['episode_time'].append(t_elapsed)
        stats['ppo_loss'].append(float(ppo_losses.get('ppo_loss', 0)))
        if curriculum:
            stats['curriculum_phase'].append(curriculum.current_phase_idx)

        # Logging
        if episode % 5 == 0:
            mean_r = np.mean(stats['episode_reward'][-10:])
            phase_str = f" phase={curriculum.current_phase.name}" if curriculum else ""
            print(
                f"[{variant}] Ep {episode:4d}: r={episode_reward:7.1f} "
                f"avg10={mean_r:7.1f} len={episode_length:3d} "
                f"loss={ppo_losses.get('ppo_loss', 0):.4f}{phase_str} "
                f"t={t_elapsed:.1f}s"
            )

        # Save best
        if len(stats['episode_reward']) >= 20:
            mean_r = np.mean(stats['episode_reward'][-20:])
            if mean_r > best_mean_reward:
                best_mean_reward = mean_r
                torch.save(policy.state_dict(),
                           os.path.join(save_dir, 'best_policy.pt'))

        # Periodic checkpoint
        if episode % 50 == 0 and episode > 0:
            torch.save({
                'episode': episode,
                'variant': variant,
                'policy_state': policy.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'stats': dict(stats),
                'curriculum': curriculum.get_state() if curriculum else None,
            }, os.path.join(save_dir, f'checkpoint_{episode}.pt'))

    # Final save
    torch.save(policy.state_dict(), os.path.join(save_dir, 'final_policy.pt'))

    # Save training stats
    stats_serializable = {k: [float(x) if isinstance(x, (np.floating, float)) else int(x) for x in v]
                          for k, v in stats.items()}
    with open(os.path.join(save_dir, 'training_stats.json'), 'w') as f:
        json.dump(stats_serializable, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training {variant} complete!")
    print(f"  Best avg reward (20-ep): {best_mean_reward:.2f}")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Saved to: {save_dir}")
    print(f"{'='*60}\n")

    return stats


def _ppo_update_no_theory(policy, optimizer, data, cfg):
    """PPO update WITHOUT theory-guided regularization (for pure-rl)."""
    n_samples = len(data['obs'])
    batch_size = min(cfg.ppo_batch_size, n_samples)
    losses_log = {}

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

            eval_out = policy.evaluate_actions(
                batch_obs, batch_discrete, batch_cont, batch_p2p, batch_mask,
            )

            # PPO clipped objective
            ratio = torch.exp(eval_out['log_prob'] - batch_old_lp)
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1 - cfg.ppo_clip_eps,
                                1 + cfg.ppo_clip_eps) * batch_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (eval_out['value'] - batch_returns).pow(2).mean()
            entropy_loss = -eval_out['entropy'].mean()

            total_loss = (policy_loss +
                          cfg.ppo_value_coef * value_loss +
                          cfg.ppo_entropy_coef * entropy_loss)

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.ppo_max_grad_norm)
            optimizer.step()

            losses_log = {
                'ppo_loss': float(total_loss.item()),
                'value_loss': float(value_loss.item()),
                'policy_loss': float(policy_loss.item()),
                'theory_loss': 0.0,
            }

    return losses_log


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RL tactical variant')
    parser.add_argument('--variant', type=str, required=True,
                        choices=['A-oursrl', 'oursrl', 'pure-rl'],
                        help='Which variant to train')
    parser.add_argument('--total-episodes', type=int, default=600)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--scenario', type=str, default='scenario_a')
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--demo-path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_variant(
        variant=args.variant,
        total_episodes=args.total_episodes,
        max_steps=args.max_steps,
        save_dir=args.save_dir,
        scenario_name=args.scenario,
        demo_path=args.demo_path,
        seed=args.seed,
    )

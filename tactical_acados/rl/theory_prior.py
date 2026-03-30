"""
Theory-guided prior computation.

Computes factorized prior:
  π_th(a|o) = π_th^d(a^d|o) * π_th^c(a^c|a^d,o)

Provides loss functions for PPO regularization:
  L_prior^d = KL(π_θ^d || π_th^d)
  L_prior^c = ||μ_θ^c - c_th*||^2
  L_game = (G_ψ(o,a) - V_e(a|o))^2
"""

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TacticalConfig, DEFAULT_CONFIG
from tactical_action import NUM_DISCRETE_ACTIONS


class TheoryPrior:
    """
    Theory-guided prior for hybrid PPO regularization.
    
    Discrete prior: Boltzmann distribution over tactical game values
    Continuous prior: heuristic continuous targets per discrete candidate
    """

    def __init__(self, cfg: TacticalConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        # Continuous targets per discrete tactic (from heuristic policy)
        # [alpha, rho_v, rho_n, rho_s, rho_w]
        self.continuous_targets = {
            0: [0.7, 0.05, 0.0, 1.0, 1.0],     # follow_center
            1: [0.8, 0.15, 0.8, 0.9, 1.3],      # overtake_left
            2: [0.8, 0.15, -0.8, 0.9, 1.3],     # overtake_right
            3: [0.6, 0.0, 0.5, 1.1, 1.5],       # defend_left
            4: [0.6, 0.0, -0.5, 1.1, 1.5],      # defend_right
            5: [0.3, -0.1, 0.0, 1.3, 1.5],      # recover_center
        }

    def compute_discrete_prior(self, game_values, safe_mask):
        """
        Compute Boltzmann discrete prior from game values.
        
        Args:
            game_values: (batch, NUM_DISCRETE) tensor of G_d values
            safe_mask: (batch, NUM_DISCRETE) binary mask
            
        Returns:
            prior_probs: (batch, NUM_DISCRETE) probability distribution
        """
        tau = self.cfg.tau_d
        logits = game_values / tau

        # Mask invalid actions
        logits = logits + (1.0 - safe_mask) * (-1e8)

        prior = F.softmax(logits, dim=-1)
        return prior

    def compute_continuous_target(self, discrete_actions):
        """
        Get continuous prior target for given discrete actions.
        
        Args:
            discrete_actions: (batch,) tensor of discrete action indices
            
        Returns:
            targets: (batch, 5) tensor
        """
        batch_size = discrete_actions.shape[0]
        targets = torch.zeros(batch_size, 5)
        for i in range(batch_size):
            d = int(discrete_actions[i].item())
            targets[i] = torch.tensor(self.continuous_targets.get(d, [0.5, 0.0, 0.0, 1.0, 1.0]))
        return targets

    def discrete_prior_loss(self, policy_probs, prior_probs):
        """
        KL divergence: D_KL(π_θ^d || π_th^d).
        
        Args:
            policy_probs: (batch, NUM_DISCRETE) from policy
            prior_probs: (batch, NUM_DISCRETE) from theory prior
        """
        # Add small epsilon for numerical stability
        eps = 1e-8
        policy_probs = policy_probs + eps
        prior_probs = prior_probs + eps

        kl = (policy_probs * (torch.log(policy_probs) - torch.log(prior_probs))).sum(dim=-1)
        return kl.mean()

    def continuous_prior_loss(self, policy_mean, target, W_c=None):
        """
        Continuous prior loss: ||μ_θ^c - c_th*||^2_W.
        
        Args:
            policy_mean: (batch, 5) predicted continuous mean
            target: (batch, 5) theory target
            W_c: optional (5,) weight vector
        """
        diff = policy_mean - target
        if W_c is not None:
            diff = diff * W_c.unsqueeze(0)
        return (diff ** 2).sum(dim=-1).mean()

    def game_value_loss(self, predicted_game_value, target_game_value):
        """
        Auxiliary game-value head loss: (G_ψ - V_e)^2.
        
        Args:
            predicted_game_value: (batch,) from network
            target_game_value: (batch,) from game value computation
        """
        return F.mse_loss(predicted_game_value, target_game_value)

    def compute_all_losses(
            self,
            policy_probs,
            cont_mean,
            discrete_actions,
            game_value_pred,
            game_value_target,
            safe_mask,
            game_values_all=None,
    ):
        """
        Compute all theory-guided regularization losses.
        
        Returns:
            dict with L_prior_d, L_prior_c, L_game, and weighted total
        """
        # Discrete prior
        if game_values_all is not None:
            prior_probs = self.compute_discrete_prior(game_values_all, safe_mask)
        else:
            # Uniform prior if no game values available
            prior_probs = safe_mask / safe_mask.sum(dim=-1, keepdim=True).clamp(min=1)

        L_prior_d = self.discrete_prior_loss(policy_probs, prior_probs)

        # Continuous prior target
        cont_target = self.compute_continuous_target(discrete_actions)
        if cont_mean.is_cuda:
            cont_target = cont_target.cuda()
        L_prior_c = self.continuous_prior_loss(cont_mean, cont_target)

        # Game value
        L_game = self.game_value_loss(game_value_pred, game_value_target)

        # Weighted total
        total = (
            self.cfg.lambda_d * L_prior_d +
            self.cfg.lambda_c * L_prior_c +
            self.cfg.lambda_g * L_game
        )

        return {
            'L_prior_d': L_prior_d,
            'L_prior_c': L_prior_c,
            'L_game': L_game,
            'L_theory_total': total,
        }

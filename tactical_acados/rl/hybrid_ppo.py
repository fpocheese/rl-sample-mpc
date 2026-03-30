"""
Hybrid PPO policy network for tactical racing.

Architecture:
  Shared encoder → {
    Discrete head:    logits for 6 tactical candidates → Categorical
    Continuous head:  mean + log_std for (α, ρ_v, ρ_n, ρ_s, ρ_w) → Gaussian
    P2P head:         single logit → Bernoulli
    Value head:       state value V(s)
    Game-value head:  auxiliary G_ψ(o, a)
  }

Safe action masking is applied to discrete logits before sampling.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical, Normal, Bernoulli
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TacticalConfig, DEFAULT_CONFIG
from tactical_action import NUM_DISCRETE_ACTIONS


def _build_mlp(input_dim, hidden_sizes, output_dim, activate_last=False):
    """Build a simple MLP."""
    layers = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    if activate_last:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class HybridPPOPolicy(nn.Module):
    """
    Hybrid PPO policy with:
    - Factorized discrete + continuous + binary action heads
    - Value head for PPO critic
    - Auxiliary game-value head for theory regularization
    """

    def __init__(
            self,
            obs_dim: int,
            cfg: TacticalConfig = DEFAULT_CONFIG,
    ):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.n_discrete = NUM_DISCRETE_ACTIONS
        self.n_continuous = 5  # alpha, rho_v, rho_n, rho_s, rho_w
        hidden = cfg.hidden_sizes

        # Shared encoder
        self.encoder = _build_mlp(obs_dim, hidden, hidden[-1], activate_last=True)

        # Discrete tactical head
        self.discrete_head = nn.Linear(hidden[-1], self.n_discrete)

        # Continuous tactical head (outputs mean)
        self.continuous_mean = nn.Linear(hidden[-1], self.n_continuous)
        # Learnable log std
        self.continuous_log_std = nn.Parameter(
            torch.zeros(self.n_continuous) - 0.5
        )

        # P2P head (Bernoulli)
        self.p2p_head = nn.Linear(hidden[-1], 1)

        # Value head (PPO critic)
        self.value_head = _build_mlp(obs_dim, [hidden[-1]], 1)

        # Auxiliary game-value head
        # Input: obs + action encoding
        action_encoding_dim = self.n_discrete + self.n_continuous + 1
        self.game_value_head = _build_mlp(
            hidden[-1] + action_encoding_dim, [128], 1,
        )

        # Action bounds for clamping continuous outputs
        self.register_buffer('cont_low', torch.tensor([
            cfg.aggressiveness_range[0],
            cfg.rho_v_range[0], cfg.rho_n_range[0],
            cfg.rho_s_range[0], cfg.rho_w_range[0],
        ], dtype=torch.float32))
        self.register_buffer('cont_high', torch.tensor([
            cfg.aggressiveness_range[1],
            cfg.rho_v_range[1], cfg.rho_n_range[1],
            cfg.rho_s_range[1], cfg.rho_w_range[1],
        ], dtype=torch.float32))

    def forward(self, obs, safe_mask=None):
        """
        Full forward pass.
        
        Args:
            obs: (batch, obs_dim) tensor
            safe_mask: (batch, n_discrete) binary mask, 1=valid
            
        Returns dict with all outputs.
        """
        features = self.encoder(obs)

        # Discrete logits with safe masking
        discrete_logits = self.discrete_head(features)
        if safe_mask is not None:
            # Set masked actions to very negative logits
            discrete_logits = discrete_logits + (1.0 - safe_mask) * (-1e8)

        # Continuous mean (scaled to bounds)
        cont_raw = self.continuous_mean(features)
        cont_mean = self.cont_low + (self.cont_high - self.cont_low) * torch.sigmoid(cont_raw)
        cont_std = torch.exp(self.continuous_log_std).expand_as(cont_mean)

        # P2P logit
        p2p_logit = self.p2p_head(features).squeeze(-1)

        # Value
        value = self.value_head(obs).squeeze(-1)

        return {
            'features': features,
            'discrete_logits': discrete_logits,
            'cont_mean': cont_mean,
            'cont_std': cont_std,
            'p2p_logit': p2p_logit,
            'value': value,
        }

    def get_action_and_value(self, obs, safe_mask=None, deterministic=False):
        """
        Sample action and compute value + log_prob.
        Used during rollout collection.
        """
        out = self.forward(obs, safe_mask)

        # Discrete action
        discrete_dist = Categorical(logits=out['discrete_logits'])
        if deterministic:
            discrete_action = out['discrete_logits'].argmax(dim=-1)
        else:
            discrete_action = discrete_dist.sample()
        discrete_log_prob = discrete_dist.log_prob(discrete_action)

        # Continuous action
        cont_dist = Normal(out['cont_mean'], out['cont_std'])
        if deterministic:
            cont_action = out['cont_mean']
        else:
            cont_action = cont_dist.sample()
        # Clamp to bounds
        cont_action = torch.clamp(cont_action, self.cont_low, self.cont_high)
        cont_log_prob = cont_dist.log_prob(cont_action).sum(dim=-1)

        # P2P action
        p2p_dist = Bernoulli(logits=out['p2p_logit'])
        if deterministic:
            p2p_action = (out['p2p_logit'] > 0).float()
        else:
            p2p_action = p2p_dist.sample()
        p2p_log_prob = p2p_dist.log_prob(p2p_action)

        # Total log prob
        total_log_prob = discrete_log_prob + cont_log_prob + p2p_log_prob

        return {
            'discrete_action': discrete_action,
            'cont_action': cont_action,
            'p2p_action': p2p_action,
            'log_prob': total_log_prob,
            'value': out['value'],
            'discrete_logits': out['discrete_logits'],
            'cont_mean': out['cont_mean'],
            'cont_std': out['cont_std'],
            'features': out['features'],
        }

    def evaluate_actions(self, obs, discrete_action, cont_action, p2p_action,
                          safe_mask=None):
        """
        Evaluate log_prob and entropy for given actions.
        Used during PPO update.
        """
        out = self.forward(obs, safe_mask)

        # Discrete
        discrete_dist = Categorical(logits=out['discrete_logits'])
        discrete_log_prob = discrete_dist.log_prob(discrete_action)
        discrete_entropy = discrete_dist.entropy()

        # Continuous
        cont_dist = Normal(out['cont_mean'], out['cont_std'])
        cont_log_prob = cont_dist.log_prob(cont_action).sum(dim=-1)
        cont_entropy = cont_dist.entropy().sum(dim=-1)

        # P2P
        p2p_dist = Bernoulli(logits=out['p2p_logit'])
        p2p_log_prob = p2p_dist.log_prob(p2p_action)
        p2p_entropy = p2p_dist.entropy()

        total_log_prob = discrete_log_prob + cont_log_prob + p2p_log_prob
        total_entropy = discrete_entropy + cont_entropy + p2p_entropy

        return {
            'log_prob': total_log_prob,
            'entropy': total_entropy,
            'value': out['value'],
            'discrete_probs': F.softmax(out['discrete_logits'], dim=-1),
            'cont_mean': out['cont_mean'],
            'features': out['features'],
        }

    def compute_game_value(self, features, action_encoding):
        """Compute auxiliary game value G_ψ(o, a)."""
        x = torch.cat([features, action_encoding], dim=-1)
        return self.game_value_head(x).squeeze(-1)

    def encode_action(self, discrete_action, cont_action, p2p_action):
        """Encode action as a vector for game-value head input."""
        batch_size = discrete_action.shape[0]
        # One-hot discrete
        discrete_onehot = F.one_hot(
            discrete_action.long(), self.n_discrete
        ).float()
        return torch.cat([discrete_onehot, cont_action,
                          p2p_action.unsqueeze(-1)], dim=-1)

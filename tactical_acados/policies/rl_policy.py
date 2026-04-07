"""
RL Policy Wrapper.

Loads a trained HybridPPOPolicy and presents the same interface as
HeuristicTacticalPolicy so it can be used in sim_tactical.py and benchmarks.
"""

import torch
import numpy as np
import os
import sys
from typing import Optional

dir_path = os.path.dirname(os.path.abspath(__file__))
tactical_dir = os.path.join(dir_path, '..')
sys.path.insert(0, os.path.join(dir_path, '..'))
sys.path.insert(0, os.path.join(dir_path, '..', 'rl'))

from rl.hybrid_ppo import HybridPPOPolicy
from tactical_action import (
    TacticalAction, DiscreteTactic, PreferenceVector,
    NUM_DISCRETE_ACTIONS,
)
from observation import TacticalObservation
from config import TacticalConfig, DEFAULT_CONFIG
from safe_wrapper import SafeTacticalWrapper

_TACTIC_TO_CARVER_MODE = {
    DiscreteTactic.FOLLOW_CENTER: 'follow',
    DiscreteTactic.OVERTAKE_LEFT: 'overtake',
    DiscreteTactic.OVERTAKE_RIGHT: 'overtake',
    DiscreteTactic.DEFEND_LEFT: 'defend',
    DiscreteTactic.DEFEND_RIGHT: 'defend',
    DiscreteTactic.PREPARE_OVERTAKE_LEFT: 'shadow',
    DiscreteTactic.PREPARE_OVERTAKE_RIGHT: 'shadow',
    DiscreteTactic.RACE_LINE: 'raceline',
}

_TACTIC_TO_SIDE = {
    DiscreteTactic.FOLLOW_CENTER: None,
    DiscreteTactic.OVERTAKE_LEFT: 'left',
    DiscreteTactic.OVERTAKE_RIGHT: 'right',
    DiscreteTactic.DEFEND_LEFT: 'left',
    DiscreteTactic.DEFEND_RIGHT: 'right',
    DiscreteTactic.PREPARE_OVERTAKE_LEFT: 'left',
    DiscreteTactic.PREPARE_OVERTAKE_RIGHT: 'right',
    DiscreteTactic.RACE_LINE: None,
}


class RLTacticalPolicy:
    """Wraps trained HybridPPOPolicy matching HeuristicTacticalPolicy interface."""

    def __init__(self, model_path, cfg=DEFAULT_CONFIG, deterministic=True, variant_name='rl'):
        self.cfg = cfg
        self.deterministic = deterministic
        self.variant_name = variant_name
        self.safe_wrapper = SafeTacticalWrapper(cfg)

        obs_dim = TacticalObservation.obs_dim(cfg)
        self.model = HybridPPOPolicy(obs_dim=obs_dim, cfg=cfg)

        if os.path.exists(model_path):
            print(f"RLPolicy[{variant_name}]: Loading from {model_path}")
            state_dict = torch.load(model_path, map_location='cpu')
            if 'policy_state' in state_dict:
                self.model.load_state_dict(state_dict['policy_state'])
            else:
                self.model.load_state_dict(state_dict)
        else:
            print(f"WARNING: {model_path} not found! Random init.")

        self.model.eval()
        self._overtake_ready_ext = False
        self._carver_mode_str = 'raceline'
        self._carver_side = None
        self.debug_info = {}

    @property
    def carver_mode_str(self):
        return self._carver_mode_str

    @property
    def carver_side(self):
        return self._carver_side

    def set_overtake_ready(self, ready):
        self._overtake_ready_ext = ready

    def reset(self):
        self._overtake_ready_ext = False
        self._carver_mode_str = 'raceline'
        self._carver_side = None
        self.debug_info = {}

    def act(self, obs):
        obs_arr = obs.to_array(self.cfg)
        obs_tensor = torch.FloatTensor(obs_arr).unsqueeze(0)
        safe_mask = self.safe_wrapper.get_safe_mask(obs)
        mask_tensor = torch.FloatTensor(safe_mask).unsqueeze(0)

        with torch.no_grad():
            action_out = self.model.get_action_and_value(
                obs_tensor, mask_tensor, deterministic=self.deterministic,
            )

        discrete_idx = int(action_out['discrete_action'].item())
        cont_vals = action_out['cont_action'].squeeze(0).numpy()
        p2p_val = bool(action_out['p2p_action'].item() > 0.5)

        tactic = DiscreteTactic(discrete_idx)
        action = TacticalAction(
            discrete_tactic=tactic,
            aggressiveness=float(cont_vals[0]),
            preference=PreferenceVector(
                rho_v=float(cont_vals[1]),
                rho_n=float(cont_vals[2]),
                rho_s=float(cont_vals[3]),
                rho_w=float(cont_vals[4]),
            ),
            p2p_trigger=p2p_val,
        )

        action = self.safe_wrapper.sanitize(action, obs)
        self._carver_mode_str = _TACTIC_TO_CARVER_MODE.get(tactic, 'follow')
        self._carver_side = _TACTIC_TO_SIDE.get(tactic, None)

        probs = torch.softmax(action_out['discrete_logits'], dim=-1).squeeze().numpy()
        self.debug_info = {
            'phase': self._carver_mode_str.upper(),
            'tactic': tactic.name,
            'discrete_probs': {DiscreteTactic(i).name: f"{p:.3f}" for i, p in enumerate(probs)},
            'aggressiveness': f"{action.aggressiveness:.3f}",
            'value': f"{action_out['value'].item():.3f}",
            'variant': self.variant_name,
        }
        return action


def load_rl_policy(variant='A-oursrl', checkpoint_dir=None, cfg=DEFAULT_CONFIG, use_best=True):
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(tactical_dir, 'checkpoints', variant.replace('-', '_'))
    filename = 'best_policy.pt' if use_best else 'final_policy.pt'
    model_path = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model at {model_path}")
    return RLTacticalPolicy(model_path=model_path, cfg=cfg, deterministic=True, variant_name=variant)

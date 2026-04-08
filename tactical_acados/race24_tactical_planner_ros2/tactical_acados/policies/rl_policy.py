"""
RL tactical policy wrapper for simulation.
Loads a trained HybridPPOPolicy and provides an 'act' method.
"""

import torch
import numpy as np
import os
import sys

# Ensure imports work
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(dir_path, '..'))
sys.path.insert(0, os.path.join(dir_path, '..', 'rl'))

from rl.hybrid_ppo import HybridPPOPolicy
from tactical_action import TacticalAction, DiscreteTactic, PreferenceVector
from observation import TacticalObservation
from config import TacticalConfig, DEFAULT_CONFIG
from safe_wrapper import SafeTacticalWrapper

class RLTacticalPolicy:
    """
    Policy that uses a trained HybridPPO model to make decisions.
    """

    def __init__(self, model_path: str, cfg: TacticalConfig = DEFAULT_CONFIG, deterministic: bool = True):
        self.cfg = cfg
        self.deterministic = deterministic
        self.safe_wrapper = SafeTacticalWrapper(cfg)

        # Load model
        obs_dim = TacticalObservation.obs_dim(cfg)
        self.model = HybridPPOPolicy(obs_dim=obs_dim, cfg=cfg)
        
        if os.path.exists(model_path):
            print(f"RLPolicy: Loading model from {model_path}")
            state_dict = torch.load(model_path, map_location='cpu')
            # Handle both full checkpoint and state_dict only
            if 'policy_state' in state_dict:
                self.model.load_state_dict(state_dict['policy_state'])
            else:
                self.model.load_state_dict(state_dict)
        else:
            print(f"WARNING: RL model file {model_path} not found! Using random initialization.")
            
        self.model.eval()

    def act(self, obs: TacticalObservation) -> TacticalAction:
        """
        Convert TacticalObservation to action using the trained model.
        """
        # 1. Prepare observation tensor
        obs_arr = obs.to_array(self.cfg)
        obs_tensor = torch.FloatTensor(obs_arr).unsqueeze(0)

        # 2. Prepare safe mask
        safe_mask = self.safe_wrapper.get_safe_mask(obs)
        mask_tensor = torch.FloatTensor(safe_mask).unsqueeze(0)

        # 3. Inference
        with torch.no_grad():
            action_out = self.model.get_action_and_value(
                obs_tensor, mask_tensor, deterministic=self.deterministic
            )

        # 4. Map output back to TacticalAction
        discrete_idx = int(action_out['discrete_action'].item())
        cont_vals = action_out['cont_action'].squeeze(0).numpy()
        p2p_val = bool(action_out['p2p_action'].item() > 0.5)

        action = TacticalAction()
        action.discrete_tactic = DiscreteTactic(discrete_idx)
        action.aggressiveness = float(cont_vals[0])
        action.preference = PreferenceVector(
            rho_v=float(cont_vals[1]),
            rho_n=float(cont_vals[2]),
            rho_s=float(cont_vals[3]),
            rho_w=float(cont_vals[4])
        )
        action.p2p_trigger = p2p_val

        # 5. Sanitize (important for feasibility)
        sanitized = self.safe_wrapper.sanitize(action, obs)
        return sanitized

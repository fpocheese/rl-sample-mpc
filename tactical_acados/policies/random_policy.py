"""
Random tactical policy.

Samples uniformly from the safe action space.
Critical test: even random policy must not break planner feasibility.
"""

import numpy as np

from tactical_action import (
    TacticalAction, DiscreteTactic, PreferenceVector,
)
from observation import TacticalObservation
from safe_wrapper import SafeTacticalWrapper
from config import TacticalConfig, DEFAULT_CONFIG


class RandomTacticalPolicy:
    """
    Random policy that samples uniformly over the safe tactical action space.
    
    All outputs go through the safe wrapper, so the planner must always
    receive feasible inputs regardless of random sampling.
    """

    def __init__(self, cfg: TacticalConfig = DEFAULT_CONFIG, seed: int = 42):
        self.cfg = cfg
        self.safe_wrapper = SafeTacticalWrapper(cfg)
        self.rng = np.random.RandomState(seed)

    def act(self, obs: TacticalObservation) -> TacticalAction:
        """Sample random tactical action from safe set."""
        # Get safe discrete set and sample uniformly
        safe_set = self.safe_wrapper.get_safe_discrete_set(obs)
        discrete_idx = self.rng.randint(0, len(safe_set))
        discrete_tactic = safe_set[discrete_idx]

        # Sample continuous parameters uniformly within bounds
        alpha = self.rng.uniform(*self.cfg.aggressiveness_range)
        rho_v = self.rng.uniform(*self.cfg.rho_v_range)
        rho_n = self.rng.uniform(*self.cfg.rho_n_range)
        rho_s = self.rng.uniform(*self.cfg.rho_s_range)
        rho_w = self.rng.uniform(*self.cfg.rho_w_range)

        # P2P: small random chance
        p2p = self.rng.random() < 0.05 and obs.p2p_available

        action = TacticalAction(
            discrete_tactic=discrete_tactic,
            aggressiveness=alpha,
            preference=PreferenceVector(rho_v, rho_n, rho_s, rho_w),
            p2p_trigger=p2p,
        )

        # MUST go through safe wrapper
        return self.safe_wrapper.sanitize(action, obs)

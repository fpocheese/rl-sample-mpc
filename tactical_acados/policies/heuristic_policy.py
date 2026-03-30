"""
Heuristic tactical policy.

Rule-based policy that generates reasonable tactical decisions based on
track geometry and opponent positions. Used as:
- Baseline for comparison
- Opponent AI policy
- Theory-guided prior target
"""

import numpy as np
from typing import Optional

from tactical_action import (
    TacticalAction, DiscreteTactic, PreferenceVector,
    TacticalMode, LateralIntention,
)
from observation import TacticalObservation
from safe_wrapper import SafeTacticalWrapper
from config import TacticalConfig, DEFAULT_CONFIG


class HeuristicTacticalPolicy:
    """
    Rule-based tactical policy for racing.
    
    Decision logic:
    1. No opponents nearby → follow_center, moderate aggressiveness
    2. Opponent ahead within gap → analyze corridors, choose wider side to overtake
    3. Opponent behind and closing → defend on raceline side
    4. High curvature zone → recover_center
    """

    def __init__(self, cfg: TacticalConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self.safe_wrapper = SafeTacticalWrapper(cfg)

    def act(self, obs: TacticalObservation) -> TacticalAction:
        """Generate tactical action from observation."""
        has_nearby = any(abs(opp.delta_s) < 60.0 for opp in obs.opponents)

        if not has_nearby:
            action = self._race_line_action(obs)
        # High curvature → conservative follow (only if battling)
        elif obs.upcoming_max_curvature > 0.025:
            action = self._conservative_action(obs)
        else:
            # Find most relevant opponent
            ahead = [o for o in obs.opponents if o.delta_s < 0]
            behind = [o for o in obs.opponents if o.delta_s > 0]

            if ahead and abs(ahead[0].delta_s) < 80.0:
                action = self._overtake_decision(obs, ahead[0])
            elif behind and abs(behind[0].delta_s) < 40.0 and behind[0].delta_V < -5.0:
                action = self._defend_decision(obs, behind[0])
            else:
                action = self._race_line_action(obs)

        # Always sanitize through safe wrapper
        return self.safe_wrapper.sanitize(action, obs)

    def _race_line_action(self, obs: TacticalObservation) -> TacticalAction:
        """No relevant opponents → pure racing line untouched."""
        return TacticalAction(
            discrete_tactic=DiscreteTactic.RACE_LINE,
            aggressiveness=1.0,
            preference=PreferenceVector(
                rho_v=0.0,
                rho_n=0.0,
                rho_s=1.0,
                rho_w=1.0,
            ),
            p2p_trigger=False,
        )

    def _conservative_action(self, obs: TacticalObservation) -> TacticalAction:
        """High curvature → conservative following."""
        return TacticalAction(
            discrete_tactic=DiscreteTactic.FOLLOW_CENTER,
            aggressiveness=0.3,
            preference=PreferenceVector(
                rho_v=-0.1,
                rho_n=0.0,
                rho_s=1.3,
                rho_w=1.5,
            ),
            p2p_trigger=False,
        )

    def _overtake_decision(self, obs: TacticalObservation,
                            target: 'OpponentState') -> TacticalAction:
        """Decide how to overtake opponent ahead."""
        gap = abs(target.delta_s)

        # If too far, just follow aggressively
        if gap > 35.0:
            return TacticalAction(
                discrete_tactic=DiscreteTactic.FOLLOW_CENTER,
                aggressiveness=0.85,
                preference=PreferenceVector(
                    rho_v=0.05,
                    rho_n=0.0,
                    rho_s=1.0,
                    rho_w=1.0,
                ),
            )

        # Analyze corridor widths
        left_space = obs.w_left - obs.ego_n
        right_space = abs(obs.w_right) + obs.ego_n
        opp_n = target.n

        # Determine which side has more room
        left_clear = left_space > self.cfg.overtake_min_corridor
        right_clear = right_space > self.cfg.overtake_min_corridor

        if left_clear and (not right_clear or left_space > right_space):
            tactic = DiscreteTactic.OVERTAKE_LEFT if gap < 15.0 else DiscreteTactic.PREPARE_OVERTAKE_LEFT
            rho_n = min(1.0, left_space * 0.3) if gap < 15.0 else min(0.8, left_space * 0.3)
        elif right_clear:
            tactic = DiscreteTactic.OVERTAKE_RIGHT if gap < 15.0 else DiscreteTactic.PREPARE_OVERTAKE_RIGHT
            rho_n = max(-1.0, -right_space * 0.3) if gap < 15.0 else max(-0.8, -right_space * 0.3)
        else:
            # No room → follow
            return TacticalAction(
                discrete_tactic=DiscreteTactic.FOLLOW_CENTER,
                aggressiveness=0.5,
                preference=PreferenceVector(rho_v=-0.05, rho_s=1.2),
            )

        # Aggressiveness based on speed difference and gap
        alpha = np.clip(0.5 + (obs.ego_V - target.V) / 20.0, 0.6, 0.95)

        # Consider P2P for close overtakes
        use_p2p = (gap < 15.0 and obs.p2p_available and
                    obs.upcoming_max_curvature < 0.015)

        return TacticalAction(
            discrete_tactic=tactic,
            aggressiveness=float(alpha),
            preference=PreferenceVector(
                rho_v=0.15,
                rho_n=rho_n,
                rho_s=0.9,
                rho_w=1.3,
            ),
            p2p_trigger=use_p2p,
        )

    def _defend_decision(self, obs: TacticalObservation,
                          threat: 'OpponentState') -> TacticalAction:
        """Decide how to defend against opponent behind."""
        # Defend on the side where opponent is likely to attack
        if threat.delta_n > 0:
            # Threat is to ego's left → defend left
            tactic = DiscreteTactic.DEFEND_LEFT
            rho_n = 0.5
        else:
            tactic = DiscreteTactic.DEFEND_RIGHT
            rho_n = -0.5

        return TacticalAction(
            discrete_tactic=tactic,
            aggressiveness=0.6,
            preference=PreferenceVector(
                rho_v=0.0,
                rho_n=rho_n,
                rho_s=1.1,
                rho_w=1.5,
            ),
            p2p_trigger=False,
        )

    def get_continuous_target(
            self,
            discrete_tactic: DiscreteTactic,
            obs: TacticalObservation,
    ) -> np.ndarray:
        """
        Get theory-guided continuous parameter target for a given discrete tactic.
        Used by the theory prior in RL training.
        
        Returns array: [alpha, rho_v, rho_n, rho_s, rho_w]
        """
        # Build a temporary action and extract continuous params
        temp_obs = obs
        if discrete_tactic == DiscreteTactic.FOLLOW_CENTER:
            return np.array([0.7, 0.05, 0.0, 1.0, 1.0])
        elif discrete_tactic == DiscreteTactic.RACE_LINE:
            return np.array([1.0, 0.0, 0.0, 1.0, 1.0])
        elif discrete_tactic == DiscreteTactic.OVERTAKE_LEFT:
            return np.array([0.8, 0.15, 0.8, 0.9, 1.3])
        elif discrete_tactic == DiscreteTactic.OVERTAKE_RIGHT:
            return np.array([0.8, 0.15, -0.8, 0.9, 1.3])
        elif discrete_tactic == DiscreteTactic.DEFEND_LEFT:
            return np.array([0.6, 0.0, 0.5, 1.1, 1.5])
        elif discrete_tactic == DiscreteTactic.DEFEND_RIGHT:
            return np.array([0.6, 0.0, -0.5, 1.1, 1.5])
        elif discrete_tactic == DiscreteTactic.PREPARE_OVERTAKE_LEFT:
            return np.array([0.75, 0.1, 0.8, 1.0, 1.2])
        elif discrete_tactic == DiscreteTactic.PREPARE_OVERTAKE_RIGHT:
            return np.array([0.75, 0.1, -0.8, 1.0, 1.2])
        else:
            return np.array([0.5, 0.0, 0.0, 1.0, 1.0])

"""
Safe tactical wrapper.

Ensures that RL/policy outputs never break planner feasibility.
All tactical actions must pass through this wrapper before reaching the planner.

Pipeline: mask discrete → clip continuous → validate → fallback if needed.
"""

import numpy as np
from typing import List, Tuple, Set

from tactical_action import (
    TacticalAction, DiscreteTactic, PreferenceVector,
    TacticalMode, LateralIntention, NUM_DISCRETE_ACTIONS,
    get_fallback_action,
)
from observation import TacticalObservation, OpponentState
from p2p import PushToPass
from config import TacticalConfig, DEFAULT_CONFIG


class SafeTacticalWrapper:
    """
    Enforces safety constraints on tactical actions.
    
    Priority order (from requirements):
    1. Planner feasibility
    2. Vehicle drivability  
    3. Safety
    4. Tactical aggressiveness
    5. RL freedom
    """

    def __init__(self, cfg: TacticalConfig = DEFAULT_CONFIG):
        self.cfg = cfg

    def get_safe_discrete_set(
            self,
            obs: TacticalObservation,
    ) -> List[DiscreteTactic]:
        """
        Compute the safe discrete tactical set D_safe(o_k).
        
        A discrete tactic is safe if at least one feasible continuous
        parameter exists for it.
        """
        safe_set = []

        for tactic in DiscreteTactic:
            if self._is_discrete_tactic_feasible(tactic, obs):
                safe_set.append(tactic)

        # Ensure at least one option always remains (follow nearest car)
        if not safe_set:
            safe_set = [DiscreteTactic.FOLLOW_CENTER]

        return safe_set

    def get_safe_mask(self, obs: TacticalObservation) -> np.ndarray:
        """
        Get binary mask for discrete actions (1 = safe, 0 = masked).
        Shape: (NUM_DISCRETE_ACTIONS,)
        """
        safe_set = self.get_safe_discrete_set(obs)
        mask = np.zeros(NUM_DISCRETE_ACTIONS, dtype=np.float32)
        for tactic in safe_set:
            mask[tactic.value] = 1.0
        return mask

    def clip_continuous(
            self,
            action: TacticalAction,
            obs: TacticalObservation,
    ) -> TacticalAction:
        """Clip continuous parameters to admissible region."""
        cfg = self.cfg

        # Clip aggressiveness
        alpha = float(np.clip(action.aggressiveness, *cfg.aggressiveness_range))

        # Clip preference vector
        pref = action.preference.clip(cfg)

        # Context-dependent further restrictions
        # In high curvature: reduce aggressiveness, increase safety margin
        if obs.upcoming_max_curvature > 0.02:  # tight corner
            alpha = min(alpha, 0.7)
            pref.rho_s = max(pref.rho_s, 1.0)

        # P2P validation
        p2p = action.p2p_trigger and obs.p2p_available

        return TacticalAction(
            discrete_tactic=action.discrete_tactic,
            aggressiveness=alpha,
            preference=pref,
            p2p_trigger=p2p,
        )

    def sanitize(
            self,
            action: TacticalAction,
            obs: TacticalObservation,
    ) -> TacticalAction:
        """
        Full sanitization pipeline.
        
        1. Check if discrete tactic is in safe set
        2. If not, replace with fallback
        3. Clip continuous parameters
        4. Final validation
        """
        safe_set = self.get_safe_discrete_set(obs)

        # Step 1: Validate discrete action
        if action.discrete_tactic not in safe_set:
            # Try to find closest safe alternative
            action = self._find_closest_safe_action(action, safe_set)

        # Step 2: Clip continuous
        action = self.clip_continuous(action, obs)

        # Step 3: Final validation
        action = self._final_validation(action, obs)

        return action

    # ------ Internal methods ------

    def _is_discrete_tactic_feasible(
            self,
            tactic: DiscreteTactic,
            obs: TacticalObservation,
    ) -> bool:
        """Check if a discrete tactic is feasible given current observation."""
        cfg = self.cfg
        mode = tactic.mode
        lateral = tactic.lateral_intention

        # FOLLOW_CENTER is always feasible (fallback: follow nearest car)
        if tactic == DiscreteTactic.FOLLOW_CENTER:
            return True

        # RECOVER_CENTER: only feasible if no opponents nearby
        if tactic == DiscreteTactic.RECOVER_CENTER:
            has_nearby = any(abs(o.delta_s) < 50 for o in obs.opponents)
            return not has_nearby  # don't slow down if cars nearby

        # Overtake checks
        if mode == TacticalMode.OVERTAKE:
            # Need an opponent ahead to overtake
            has_opp_ahead = any(
                opp.delta_s < 0 and abs(opp.delta_s) < 100.0
                for opp in obs.opponents
            )
            if not has_opp_ahead:
                return False

            # Check if corridor exists on the desired side
            if lateral == LateralIntention.LEFT:
                corridor = obs.w_left - abs(obs.ego_n)
                # Check opponent doesn't block left
                opp_blocks_left = any(
                    opp.delta_s < 0 and abs(opp.delta_s) < 50.0 and opp.delta_n < -1.0
                    for opp in obs.opponents
                )
                return corridor > cfg.overtake_min_corridor and not opp_blocks_left
            elif lateral == LateralIntention.RIGHT:
                corridor = abs(obs.w_right) - abs(obs.ego_n)
                opp_blocks_right = any(
                    opp.delta_s < 0 and abs(opp.delta_s) < 50.0 and opp.delta_n > 1.0
                    for opp in obs.opponents
                )
                return corridor > cfg.overtake_min_corridor and not opp_blocks_right

        # Defend checks
        if mode == TacticalMode.DEFEND:
            # Need an opponent behind to defend against
            has_opp_behind = any(
                opp.delta_s > 0 and abs(opp.delta_s) < 50.0
                for opp in obs.opponents
            )
            if not has_opp_behind:
                return False
            # Check corridor exists on the defend side
            if lateral == LateralIntention.LEFT:
                return obs.w_left > cfg.vehicle_width / 2.0 + 0.5
            elif lateral == LateralIntention.RIGHT:
                return abs(obs.w_right) > cfg.vehicle_width / 2.0 + 0.5

        return True

    def _find_closest_safe_action(
            self,
            action: TacticalAction,
            safe_set: List[DiscreteTactic],
    ) -> TacticalAction:
        """Replace unsafe discrete tactic with closest safe alternative."""
        # Priority: same mode different lateral > same lateral different mode > fallback
        orig_mode = action.discrete_tactic.mode
        orig_lateral = action.discrete_tactic.lateral_intention

        # Try same mode, any lateral
        for t in safe_set:
            if t.mode == orig_mode:
                return TacticalAction(
                    discrete_tactic=t,
                    aggressiveness=action.aggressiveness,
                    preference=action.preference,
                    p2p_trigger=action.p2p_trigger,
                )

        # Try follow_center
        if DiscreteTactic.FOLLOW_CENTER in safe_set:
            return TacticalAction(
                discrete_tactic=DiscreteTactic.FOLLOW_CENTER,
                aggressiveness=min(action.aggressiveness, 0.5),
                preference=action.preference,
                p2p_trigger=action.p2p_trigger,
            )

        # Last resort: fallback
        return get_fallback_action()

    def _final_validation(
            self,
            action: TacticalAction,
            obs: TacticalObservation,
    ) -> TacticalAction:
        """Final safety checks after all processing."""
        # Check speed is not too aggressive for curvature
        if obs.upcoming_max_curvature > 0.03:  # very tight
            if action.aggressiveness > 0.6:
                action.aggressiveness = 0.6
            if action.preference.rho_v > 0.0:
                action.preference.rho_v = 0.0

        # Check lateral bias is within track
        max_n = obs.w_left - self.cfg.vehicle_width / 2.0 - 0.3
        min_n = obs.w_right + self.cfg.vehicle_width / 2.0 + 0.3
        action.preference.rho_n = float(np.clip(
            action.preference.rho_n, min_n - obs.ego_n, max_n - obs.ego_n
        ))

        return action

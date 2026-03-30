"""
Tactical action data structures.

Defines the hybrid tactical action space:
  a_k = (d_k, c_k)
where d_k = (mode, lateral_intention) is discrete,
and c_k = (alpha, rho, b) is continuous/binary.
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List
import numpy as np

from config import TacticalConfig, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Discrete tactical components
# ---------------------------------------------------------------------------

class TacticalMode(IntEnum):
    """Tactical mode m_k."""
    FOLLOW = 0
    OVERTAKE = 1
    DEFEND = 2
    PREPARE_OVERTAKE = 3
    SOLO = 4


class LateralIntention(IntEnum):
    """Lateral intention ℓ_k."""
    CENTER = 0
    LEFT = 1
    RIGHT = 2


class DiscreteTactic(IntEnum):
    """Combined discrete tactical candidate set D.
    Each element encodes (mode, lateral_intention).
    """
    FOLLOW_CENTER = 0
    OVERTAKE_LEFT = 1
    OVERTAKE_RIGHT = 2
    DEFEND_LEFT = 3
    DEFEND_RIGHT = 4
    PREPARE_OVERTAKE_LEFT = 5
    PREPARE_OVERTAKE_RIGHT = 6
    RACE_LINE = 7

    @property
    def mode(self) -> TacticalMode:
        return _DISCRETE_TO_MODE[self.value]

    @property
    def lateral_intention(self) -> LateralIntention:
        return _DISCRETE_TO_LATERAL[self.value]


_DISCRETE_TO_MODE = {
    0: TacticalMode.FOLLOW,
    1: TacticalMode.OVERTAKE,
    2: TacticalMode.OVERTAKE,
    3: TacticalMode.DEFEND,
    4: TacticalMode.DEFEND,
    5: TacticalMode.PREPARE_OVERTAKE,
    6: TacticalMode.PREPARE_OVERTAKE,
    7: TacticalMode.SOLO,
}
_DISCRETE_TO_LATERAL = {
    0: LateralIntention.CENTER,
    1: LateralIntention.LEFT,
    2: LateralIntention.RIGHT,
    3: LateralIntention.LEFT,
    4: LateralIntention.RIGHT,
    5: LateralIntention.LEFT,
    6: LateralIntention.RIGHT,
    7: LateralIntention.CENTER,
}

NUM_DISCRETE_ACTIONS = len(DiscreteTactic)


# ---------------------------------------------------------------------------
# Continuous tactical components
# ---------------------------------------------------------------------------

@dataclass
class PreferenceVector:
    """Continuous preference vector ρ_k."""
    rho_v: float = 0.0   # speed bias
    rho_n: float = 0.0   # terminal lateral bias [m]
    rho_s: float = 1.0   # safety-margin scale
    rho_w: float = 1.0   # interaction weight scale

    def to_array(self) -> np.ndarray:
        return np.array([self.rho_v, self.rho_n, self.rho_s, self.rho_w])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'PreferenceVector':
        return cls(rho_v=float(arr[0]), rho_n=float(arr[1]),
                   rho_s=float(arr[2]), rho_w=float(arr[3]))

    def clip(self, cfg: TacticalConfig = DEFAULT_CONFIG) -> 'PreferenceVector':
        """Clip to admissible bounds."""
        return PreferenceVector(
            rho_v=float(np.clip(self.rho_v, *cfg.rho_v_range)),
            rho_n=float(np.clip(self.rho_n, *cfg.rho_n_range)),
            rho_s=float(np.clip(self.rho_s, *cfg.rho_s_range)),
            rho_w=float(np.clip(self.rho_w, *cfg.rho_w_range)),
        )


# ---------------------------------------------------------------------------
# Complete tactical action
# ---------------------------------------------------------------------------

@dataclass
class TacticalAction:
    """Full hybrid tactical action a_k = (d_k, c_k)."""
    # Discrete branch
    discrete_tactic: DiscreteTactic = DiscreteTactic.FOLLOW_CENTER
    # Continuous branch
    aggressiveness: float = 0.5  # α_k ∈ [0, 1]
    preference: PreferenceVector = field(default_factory=PreferenceVector)
    # Binary branch
    p2p_trigger: bool = False

    @property
    def mode(self) -> TacticalMode:
        return self.discrete_tactic.mode

    @property
    def lateral_intention(self) -> LateralIntention:
        return self.discrete_tactic.lateral_intention

    def to_array(self) -> np.ndarray:
        """Flatten to numeric array for RL: [discrete_idx, alpha, rho(4), p2p]."""
        return np.concatenate([
            [float(self.discrete_tactic.value)],
            [self.aggressiveness],
            self.preference.to_array(),
            [float(self.p2p_trigger)],
        ])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'TacticalAction':
        return cls(
            discrete_tactic=DiscreteTactic(int(arr[0])),
            aggressiveness=float(arr[1]),
            preference=PreferenceVector.from_array(arr[2:6]),
            p2p_trigger=bool(arr[6] > 0.5),
        )

    def difference(self, other: 'TacticalAction') -> float:
        """Compute action difference for smoothness penalty."""
        discrete_diff = 0.0 if self.discrete_tactic == other.discrete_tactic else 1.0
        cont_diff = (
            (self.aggressiveness - other.aggressiveness) ** 2
            + np.sum((self.preference.to_array() - other.preference.to_array()) ** 2)
        )
        return discrete_diff + cont_diff


# ---------------------------------------------------------------------------
# Planner guidance (output of tactical-to-planner mapping)
# ---------------------------------------------------------------------------

@dataclass
class PlannerGuidance:
    """Guidance parameters that modify the ACADOS local planner behavior."""
    # Corridor modification
    n_bias_per_stage: Optional[np.ndarray] = None       # lateral bias array [N]
    n_left_override: Optional[np.ndarray] = None        # left bound override [N]
    n_right_override: Optional[np.ndarray] = None       # right bound override [N]
    # Terminal target
    terminal_n_target: float = 0.0                      # desired terminal lateral offset [m]
    # Speed
    speed_scale: float = 1.0                            # multiplier on V_max
    speed_cap: float = 90.0                             # absolute V_max cap [m/s]
    # Safety
    safety_distance: float = 0.5                        # [m]
    # Interaction
    interaction_weight: float = 1.0
    # Target tracking
    follow_target_id: int = -1                          # -1 = none
    # P2P
    p2p_active: bool = False

    def get_effective_v_max(self, base_v_max: float,
                            cfg: TacticalConfig = DEFAULT_CONFIG) -> float:
        """Compute effective V_max considering speed_scale and P2P."""
        v = base_v_max * self.speed_scale
        if self.p2p_active:
            v += cfg.p2p_speed_boost
        return min(v, self.speed_cap)


# ---------------------------------------------------------------------------
# Default / fallback action
# ---------------------------------------------------------------------------

def get_fallback_action() -> TacticalAction:
    """Fallback action: follow nearest car when decision fails.
    No autonomous slowdown — always keep racing, just follow safely.
    """
    return TacticalAction(
        discrete_tactic=DiscreteTactic.RACE_LINE,
        aggressiveness=1.0,
        preference=PreferenceVector(
            rho_v=0.0, rho_n=0.0, rho_s=1.0, rho_w=1.0
        ),
        p2p_trigger=False,
    )


def get_default_guidance(cfg: TacticalConfig = DEFAULT_CONFIG) -> PlannerGuidance:
    """Default guidance with no tactical modification."""
    return PlannerGuidance(
        safety_distance=cfg.safety_distance_default,
        speed_cap=90.0,
    )

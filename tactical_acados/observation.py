"""
Observation construction for the tactical layer.

Builds a structured observation from ego state, opponent states,
track geometry, and P2P status.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from config import TacticalConfig, DEFAULT_CONFIG


@dataclass
class OpponentState:
    """Observed state of a single opponent."""
    vehicle_id: int = -1
    s: float = 0.0
    n: float = 0.0
    V: float = 0.0
    chi: float = 0.0
    x: float = 0.0
    y: float = 0.0
    # Relative to ego
    delta_s: float = 0.0      # s_ego - s_opp (positive = ego ahead)
    delta_n: float = 0.0      # n_ego - n_opp
    delta_V: float = 0.0      # V_ego - V_opp
    # Prediction
    pred_s: Optional[np.ndarray] = None
    pred_n: Optional[np.ndarray] = None
    pred_x: Optional[np.ndarray] = None
    pred_y: Optional[np.ndarray] = None


@dataclass
class TacticalObservation:
    """Full observation for the tactical decision layer."""
    # Ego state
    ego_s: float = 0.0
    ego_n: float = 0.0
    ego_V: float = 0.0
    ego_chi: float = 0.0
    ego_ax: float = 0.0
    ego_ay: float = 0.0

    # Track geometry at ego position
    curvature: float = 0.0                    # Omega_z at ego s
    w_left: float = 5.0                       # left track width at ego s
    w_right: float = -5.0                     # right track width at ego s
    dist_to_next_apex: float = 100.0          # approximate distance to next turn
    upcoming_max_curvature: float = 0.0       # max curvature in next 200m

    # Opponent information
    opponents: List[OpponentState] = field(default_factory=list)

    # P2P state
    p2p_available: bool = True
    p2p_active: bool = False
    p2p_remaining_frac: float = 0.0

    # Previous tactical action (for smoothness)
    prev_discrete_tactic: int = 0
    prev_aggressiveness: float = 0.5
    prev_rho: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0, 1.0]))

    # Planner health
    planner_healthy: bool = True

    def to_array(self, cfg: TacticalConfig = DEFAULT_CONFIG) -> np.ndarray:
        """Flatten to a numeric vector for RL."""
        ego = [self.ego_s / 3000.0,  # normalize by track length
               self.ego_n / 10.0,
               self.ego_V / 90.0,
               self.ego_chi / (np.pi / 2),
               self.ego_ax / 20.0,
               self.ego_ay / 20.0]

        track = [self.curvature * 100.0,
                 self.w_left / 10.0,
                 self.w_right / 10.0,
                 self.dist_to_next_apex / 200.0,
                 self.upcoming_max_curvature * 100.0]

        # Opponent features (pad to max n_opponents)
        opp_features = []
        for i in range(cfg.n_opponents):
            if i < len(self.opponents):
                opp = self.opponents[i]
                opp_features.extend([
                    opp.delta_s / 100.0,
                    opp.delta_n / 10.0,
                    opp.delta_V / 30.0,
                    opp.V / 90.0,
                ])
            else:
                opp_features.extend([0.0, 0.0, 0.0, 0.0])

        p2p = [float(self.p2p_available),
               float(self.p2p_active),
               self.p2p_remaining_frac]

        prev_action = [self.prev_discrete_tactic / 5.0,
                       self.prev_aggressiveness]
        prev_action.extend(self.prev_rho.tolist())

        planner = [float(self.planner_healthy)]

        return np.array(
            ego + track + opp_features + p2p + prev_action + planner,
            dtype=np.float32
        )

    @staticmethod
    def obs_dim(cfg: TacticalConfig = DEFAULT_CONFIG) -> int:
        """Return observation dimension."""
        return 6 + 5 + cfg.n_opponents * 4 + 3 + 6 + 1


def build_observation(
        ego_state: dict,
        opponents: List[dict],
        track_handler,
        p2p_state: list,
        prev_action_array: np.ndarray,
        planner_healthy: bool,
        cfg: TacticalConfig = DEFAULT_CONFIG,
) -> TacticalObservation:
    """Build a TacticalObservation from raw simulation state."""
    s_ego = ego_state['s']
    s_track_len = track_handler.s[-1]

    # Track geometry at ego position
    curvature = float(np.interp(s_ego, track_handler.s, track_handler.Omega_z,
                                period=s_track_len))
    w_left = float(np.interp(s_ego, track_handler.s, track_handler.w_tr_left,
                              period=s_track_len))
    w_right = float(np.interp(s_ego, track_handler.s, track_handler.w_tr_right,
                               period=s_track_len))

    # Upcoming curvature analysis
    s_lookahead = np.linspace(s_ego, s_ego + 200.0, 50) % s_track_len
    curvatures = np.interp(s_lookahead, track_handler.s, np.abs(track_handler.Omega_z),
                           period=s_track_len)
    upcoming_max_curv = float(np.max(curvatures))

    # Find next apex (max curvature point)
    apex_idx = np.argmax(curvatures)
    dist_to_apex = float(apex_idx * 4.0)  # 200m / 50 points = 4m per point

    # Build opponent states
    opp_states = []
    for opp in opponents:
        delta_s = s_ego - opp['s']
        # Handle wrapping
        if delta_s > s_track_len / 2:
            delta_s -= s_track_len
        elif delta_s < -s_track_len / 2:
            delta_s += s_track_len

        opp_state = OpponentState(
            vehicle_id=opp.get('id', -1),
            s=opp['s'],
            n=opp['n'],
            V=opp['V'],
            chi=opp.get('chi', 0.0),
            x=opp.get('x', 0.0),
            y=opp.get('y', 0.0),
            delta_s=delta_s,
            delta_n=ego_state['n'] - opp['n'],
            delta_V=ego_state['V'] - opp['V'],
            pred_s=opp.get('pred_s'),
            pred_n=opp.get('pred_n'),
            pred_x=opp.get('pred_x'),
            pred_y=opp.get('pred_y'),
        )
        opp_states.append(opp_state)

    # Sort by absolute delta_s (closest first)
    opp_states.sort(key=lambda o: abs(o.delta_s))

    obs = TacticalObservation(
        ego_s=s_ego,
        ego_n=ego_state['n'],
        ego_V=ego_state['V'],
        ego_chi=ego_state['chi'],
        ego_ax=ego_state['ax'],
        ego_ay=ego_state['ay'],
        curvature=curvature,
        w_left=w_left,
        w_right=w_right,
        dist_to_next_apex=dist_to_apex,
        upcoming_max_curvature=upcoming_max_curv,
        opponents=opp_states,
        p2p_available=bool(p2p_state[0]),
        p2p_active=bool(p2p_state[1]),
        p2p_remaining_frac=float(p2p_state[2]),
        prev_discrete_tactic=int(prev_action_array[0]),
        prev_aggressiveness=float(prev_action_array[1]),
        prev_rho=prev_action_array[2:6],
        planner_healthy=planner_healthy,
    )
    return obs

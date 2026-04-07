"""
Central configuration for the tactical ACADOS planning system.
All parameters are defined here for easy tuning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class TacticalConfig:
    """Master configuration for the tactical planning system."""

    # ---- Online planner output format (FIXED) ----
    planning_horizon: float = 3.75        # seconds
    n_trajectory_points: int = 30         # number of output points
    dt: float = 0.125                     # seconds between points

    # ---- ACADOS local planner ----
    optimization_horizon_m: float = 300.0 # meters (s-domain horizon, used as initial/fallback)
    N_steps_acados: int = 150             # shooting nodes in ACADOS
    gg_margin: float = 0.1               # GG diagram margin
    safety_distance_default: float = 0.5  # meters



    # ---- Simulation ----
    assumed_calc_time: float = 0.125      # seconds per sim step (matches dt)
    V_min: float = 5.0                    # minimum speed m/s

    # ---- Tactical action bounds ----
    aggressiveness_range: Tuple[float, float] = (0.0, 1.0)
    rho_v_range: Tuple[float, float] = (-0.25, 0.25)    # speed bias
    rho_n_range: Tuple[float, float] = (-1.5, 1.5)      # terminal lateral bias [m]
    rho_s_range: Tuple[float, float] = (0.7, 1.5)       # safety-margin scale
    rho_w_range: Tuple[float, float] = (0.5, 2.0)       # interaction weight scale

    # ---- Reward weights ----
    w_prog: float = 1.0
    w_race: float = 1.2    # Increased (was 0.5)
    w_safe: float = 3.5    # Decreased (was 5.0)
    w_term: float = 1.0
    w_ctrl: float = 0.3
    w_p2p: float = 0.2
    w_push: float = 0.6    # New: Drafting reward
    w_side: float = 1.0    # New: Side-by-side reward

    # ---- Safety penalties ----
    collision_penalty: float = -150.0   # More severe (was -100)
    off_track_penalty: float = -50.0
    unsafe_ttc_threshold: float = 0.8   # tighter (was 1.0)
    unsafe_gap_threshold: float = 2.5   # tighter (was 3.0)

    # ---- P2P ----
    p2p_duration: float = 15.0           # seconds
    p2p_power_boost_fraction: float = 0.14  # ~50hp / 357kW ≈ 0.14
    p2p_speed_boost: float = 5.0         # m/s added to v_max during P2P

    # ---- Boltzmann temperature for theory prior ----
    tau_d: float = 1.0

    # ---- Prior regularization weights ----
    lambda_d: float = 0.1    # discrete prior KL weight
    lambda_c: float = 0.05   # continuous prior MSE weight
    lambda_g: float = 0.1    # game-value head weight

    # ---- PPO hyperparameters ----
    ppo_lr: float = 3e-4
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip_eps: float = 0.2
    ppo_entropy_coef: float = 0.01
    ppo_value_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_n_epochs: int = 4
    ppo_batch_size: int = 64
    ppo_rollout_steps: int = 256

    # ---- Network architecture ----
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])

    # ---- Opponent parameters ----
    sensor_range: float = 200.0           # meters
    n_opponents: int = 2
    opponent_speed_scale: float = 0.85    # fraction of raceline speed

    # ---- Corridor shaping ----
    corridor_safety_margin: float = 1.0   # meters around opponent
    overtake_min_corridor: float = 2.5    # minimum corridor width for overtake [m]

    # ---- Vehicle (loaded from YAML, but defaults for dallaraAV21) ----
    vehicle_width: float = 1.93
    vehicle_length: float = 5.30

    # ---- Follow module ----
    follow_gap_default: float = 6.0       # desired following gap [m], enough to allow pulling out (抽头超车)
    follow_gap_min: float = 4.0           # minimum safe following gap [m]
    follow_funnel_half: float = 1.0       # tighter funnel for following
    follow_remove_speed_cap: bool = True  # whether to ignore speed cap in follow mode
    follow_speed_match_gain: float = 0.8  # how quickly to match leader speed
    follow_virtual_wall_stiffness: float = 5.0  # penalty stiffness for virtual wall
    follow_decel_max: float = 8.0         # max deceleration for following [m/s^2]


# Default instance
DEFAULT_CONFIG = TacticalConfig()

"""
Gymnasium-compatible tactical RL environment.

Wraps the full tactical simulation pipeline into a standard RL interface:
  obs, reward, done, truncated, info = env.step(action)
"""

import numpy as np
import os
import sys
import yaml
from typing import Optional, Tuple, Dict, Any

dir_path = os.path.dirname(os.path.abspath(__file__))
tactical_dir = os.path.join(dir_path, '..')
project_root = os.path.join(tactical_dir, '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, tactical_dir)

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False

from config import TacticalConfig, DEFAULT_CONFIG
from tactical_action import (
    TacticalAction, DiscreteTactic, PreferenceVector,
    NUM_DISCRETE_ACTIONS, get_fallback_action,
)
from observation import TacticalObservation, build_observation
from safe_wrapper import SafeTacticalWrapper
from planner_guidance import TacticalToPlanner
from acados_planner import AcadosTacticalPlanner
from opponent import OpponentVehicle
from p2p import PushToPass
from sim_acados_only import load_setup, create_initial_state, perfect_tracking_update

from reward import RewardComputer


class TacticalRacingEnv:
    """
    Gymnasium-compatible tactical racing environment.
    
    Observation: float32 vector (see observation.py)
    Action: Dict with discrete (int), continuous (float[5]), p2p (int)
    
    The environment handles:
    - Scenario loading and reset
    - Observation construction
    - Action decoding and safe wrapping
    - Planner execution
    - State propagation
    - Reward computation
    - Episode termination
    """

    def __init__(
            self,
            scenario_name: str = 'scenario_a',
            cfg: TacticalConfig = DEFAULT_CONFIG,
            randomize_init: bool = True,
            max_steps: int = 500,
    ):
        self.cfg = cfg
        self.scenario_name = scenario_name
        self.randomize_init = randomize_init
        self.max_episode_steps = max_steps
        self.rng = np.random.RandomState(42)

        # Load scenario
        scenario_path = os.path.join(tactical_dir, 'scenarios',
                                      f'{scenario_name}.yml')
        with open(scenario_path, 'r') as f:
            self.scenario = yaml.safe_load(f)

        sc = self.scenario['scenario']

        # Load track/vehicle setup
        params, track_handler, gg_handler, local_planner, global_planner = load_setup(
            cfg,
            track_name=sc.get('track_name', 'yas_user_smoothed'),
            vehicle_name=sc.get('vehicle_name', 'eav25_car'),
            raceline_name=sc.get('raceline_name',
                                  'yasnorth_3d_rl_as_ref_eav25_car_gg_0.1'),
        )
        self.params = params
        self.track_handler = track_handler
        self.gg_handler = gg_handler
        self.local_planner = local_planner
        self.global_planner = global_planner

        # Create components
        self.planner = AcadosTacticalPlanner(
            local_planner, global_planner, track_handler,
            params['vehicle_params'], cfg,
        )
        self.safe_wrapper = SafeTacticalWrapper(cfg)
        self.tactical_mapper = TacticalToPlanner(track_handler, cfg)
        self.reward_computer = RewardComputer(cfg)

        # Spaces
        obs_dim = TacticalObservation.obs_dim(cfg)
        self.observation_space_shape = (obs_dim,)
        self.n_discrete_actions = NUM_DISCRETE_ACTIONS
        self.n_continuous_actions = 5  # alpha, rho_v, rho_n, rho_s, rho_w
        self.n_p2p_actions = 2

        # State
        self.ego_state = None
        self.opponents = []
        self.p2p = None
        self.prev_action = None
        self.step_count = 0
        self._setup_done = False

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment for new episode."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.planner.reset()
        self.p2p = PushToPass(self.cfg)
        self.step_count = 0
        self.prev_action = get_fallback_action()

        ego_cfg = self.scenario['ego']
        opp_cfgs = self.scenario.get('opponents', [])

        # Initialize ego with optional randomization
        s_init = ego_cfg['start_s']
        n_init = ego_cfg['start_n']
        V_init = ego_cfg['start_V']

        if self.randomize_init:
            s_init += self.rng.uniform(-20, 20)
            n_init += self.rng.uniform(-0.5, 0.5)
            V_init += self.rng.uniform(-5, 5)
            V_init = max(V_init, 10.0)

        self.ego_state = create_initial_state(
            self.track_handler, s_init, n_init, V_init,
        )

        # Initialize opponents
        self.opponents = []
        for opp_cfg in opp_cfgs:
            s_opp = opp_cfg['start_s']
            if self.randomize_init:
                s_opp += self.rng.uniform(-15, 15)

            opp = OpponentVehicle(
                vehicle_id=opp_cfg['id'],
                s_init=s_opp,
                n_init=opp_cfg.get('start_n', 0.0),
                V_init=opp_cfg.get('start_V', 40.0),
                track_handler=self.track_handler,
                global_planner=self.global_planner,
                speed_scale=opp_cfg.get('speed_scale', 0.85),
                cfg=self.cfg,
            )
            self.opponents.append(opp)

        obs = self._get_obs()
        info = {'safe_mask': self.safe_wrapper.get_safe_mask(self._last_obs)}
        return obs, info

    def step(
            self,
            action_dict: dict,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step.
        
        Args:
            action_dict: {
                'discrete': int (0-5),
                'continuous': np.ndarray shape (5,),  # alpha, rho_v, rho_n, rho_s, rho_w
                'p2p': int (0 or 1)
            }
        
        Returns:
            obs, reward, terminated, truncated, info
        """
        # Decode action
        discrete_idx = int(action_dict['discrete'])
        continuous = np.asarray(action_dict['continuous'], dtype=np.float32)
        p2p_trigger = bool(action_dict.get('p2p', 0))

        tactical_action = TacticalAction(
            discrete_tactic=DiscreteTactic(discrete_idx),
            aggressiveness=float(continuous[0]),
            preference=PreferenceVector(
                rho_v=float(continuous[1]),
                rho_n=float(continuous[2]),
                rho_s=float(continuous[3]),
                rho_w=float(continuous[4]),
            ),
            p2p_trigger=p2p_trigger,
        )

        # Safe wrapper
        tactical_action = self.safe_wrapper.sanitize(tactical_action, self._last_obs)

        # P2P
        if tactical_action.p2p_trigger and self.p2p.available:
            self.p2p.activate()
        tactical_action.p2p_trigger = self.p2p.active

        # Map to planner guidance
        guidance = self.tactical_mapper.map(
            tactical_action, self._last_obs, self.cfg.N_steps_acados,
        )

        # Plan
        prev_ego = dict(self.ego_state)
        trajectory = self.planner.plan(self.ego_state, guidance)

        # Move opponents
        prev_opponents = [opp.get_state() for opp in self.opponents]
        for opp in self.opponents:
            opp.step(self.cfg.assumed_calc_time, self.ego_state)

        # P2P step
        self.p2p.step(self.cfg.assumed_calc_time)

        # Perfect tracking
        self.ego_state = perfect_tracking_update(
            self.ego_state, trajectory, self.cfg.assumed_calc_time,
            self.track_handler,
        )

        # Reward
        curr_opponents = [opp.get_state() for opp in self.opponents]
        reward_dict = self.reward_computer.compute(
            ego_state=self.ego_state,
            ego_state_prev=prev_ego,
            opponents=curr_opponents,
            opponents_prev=prev_opponents,
            action=tactical_action,
            prev_action=self.prev_action,
            planner_healthy=self.planner.planner_healthy,
            track_handler=self.track_handler,
            p2p_active=self.p2p.active,
        )

        self.prev_action = tactical_action
        self.step_count += 1

        # Termination checks
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_episode_steps

        # Observation
        obs = self._get_obs()
        info = {
            'safe_mask': self.safe_wrapper.get_safe_mask(self._last_obs),
            'reward_components': reward_dict,
            'tactical_action': tactical_action.discrete_tactic.name,
            'planner_healthy': self.planner.planner_healthy,
        }

        return obs, float(reward_dict['total']), terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        opp_states = []
        for opp in self.opponents:
            state = opp.get_state()
            pred = opp.predict()
            state['pred_s'] = pred['pred_s']
            state['pred_n'] = pred['pred_n']
            state['pred_x'] = pred['pred_x']
            state['pred_y'] = pred['pred_y']
            opp_states.append(state)

        self._last_obs = build_observation(
            ego_state=self.ego_state,
            opponents=opp_states,
            track_handler=self.track_handler,
            p2p_state=self.p2p.get_state_vector(),
            prev_action_array=self.prev_action.to_array(),
            planner_healthy=self.planner.planner_healthy,
            cfg=self.cfg,
        )
        return self._last_obs.to_array(self.cfg)

    def _check_termination(self) -> bool:
        """Check for episode termination conditions."""
        # Off-track
        s = self.ego_state['s']
        n = self.ego_state['n']
        w_left = float(np.interp(s, self.track_handler.s,
                                  self.track_handler.w_tr_left,
                                  period=self.track_handler.s[-1]))
        w_right = float(np.interp(s, self.track_handler.s,
                                   self.track_handler.w_tr_right,
                                   period=self.track_handler.s[-1]))
        veh_half = self.cfg.vehicle_width / 2.0
        if n > w_left + veh_half or n < w_right - veh_half:
            return True

        # Collision
        for opp in self.opponents:
            dx = self.ego_state['x'] - opp.x
            dy = self.ego_state['y'] - opp.y
            dist = np.sqrt(dx**2 + dy**2)
            if dist < self.cfg.vehicle_length * 0.4:
                return True

        # Scenario boundary
        sc = self.scenario['scenario']
        s_end = sc.get('s_end', 1e6)
        if self.ego_state['s'] > s_end:
            return True

        # Speed too low (stuck)
        if self.ego_state['V'] < 1.0:
            return True

        return False

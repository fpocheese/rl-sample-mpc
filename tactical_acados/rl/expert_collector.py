"""
Expert demonstration collector.

Runs the heuristic policy through scenarios and stores (obs, action) pairs
for Behavioral Cloning (BC) pretraining of the A-oursrl variant.
"""

import os
import sys
import numpy as np
import pickle
from typing import List, Dict

dir_path = os.path.dirname(os.path.abspath(__file__))
tactical_dir = os.path.join(dir_path, '..')
project_root = os.path.join(tactical_dir, '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, tactical_dir)

from config import TacticalConfig, DEFAULT_CONFIG
from tactical_action import TacticalAction, DiscreteTactic, NUM_DISCRETE_ACTIONS
from observation import TacticalObservation, build_observation
from safe_wrapper import SafeTacticalWrapper
from planner_guidance import TacticalToPlanner
from acados_planner import AcadosTacticalPlanner
from opponent import OpponentVehicle
from p2p import PushToPass
from sim_acados_only import load_setup, create_initial_state, perfect_tracking_update

import yaml


class ExpertDemoCollector:
    """Collect demonstrations from heuristic policy."""

    def __init__(self, cfg: TacticalConfig = DEFAULT_CONFIG):
        self.cfg = cfg

    def collect_scenario(
            self,
            scenario_name: str,
            n_episodes: int = 20,
            max_steps: int = 500,
            seed_base: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Run heuristic policy on a scenario and collect transitions.

        Returns dict with keys: obs, discrete_actions, cont_actions, p2p_actions, rewards
        """
        # Lazy import to avoid circular deps
        from tactical_env import TacticalRacingEnv

        # We also need the heuristic policy
        sys.path.insert(0, os.path.join(tactical_dir, 'policies'))
        from heuristic_policy import HeuristicTacticalPolicy

        env = TacticalRacingEnv(
            scenario_name=scenario_name,
            cfg=self.cfg,
            randomize_init=True,
            max_steps=max_steps,
        )

        policy = HeuristicTacticalPolicy(cfg=self.cfg)

        all_obs = []
        all_discrete = []
        all_cont = []
        all_p2p = []
        all_rewards = []
        all_safe_masks = []

        for ep in range(n_episodes):
            obs, info = env.reset(seed=seed_base + ep)
            safe_mask = info.get('safe_mask', np.ones(NUM_DISCRETE_ACTIONS))
            if hasattr(policy, 'reset'):
                policy.reset()
            # Reset FSM state manually for heuristic
            policy.phase = "RACELINE"
            policy.phase_time = 0.0
            policy._overtake_ready_ext = False

            ep_reward = 0.0
            for step in range(max_steps):
                # Get heuristic action
                heuristic_action = policy.act(env._last_obs)

                # Convert to env action_dict
                action_dict = {
                    'discrete': heuristic_action.discrete_tactic.value,
                    'continuous': np.array([
                        heuristic_action.aggressiveness,
                        heuristic_action.preference.rho_v,
                        heuristic_action.preference.rho_n,
                        heuristic_action.preference.rho_s,
                        heuristic_action.preference.rho_w,
                    ], dtype=np.float32),
                    'p2p': int(heuristic_action.p2p_trigger),
                }

                # Store demo
                all_obs.append(obs.copy())
                all_discrete.append(action_dict['discrete'])
                all_cont.append(action_dict['continuous'].copy())
                all_p2p.append(float(action_dict['p2p']))
                all_safe_masks.append(safe_mask.copy())

                # Step
                next_obs, reward, terminated, truncated, info = env.step(action_dict)
                all_rewards.append(reward)
                ep_reward += reward

                obs = next_obs
                safe_mask = info.get('safe_mask', np.ones(NUM_DISCRETE_ACTIONS))

                if terminated or truncated:
                    break

            print(f"  Expert ep {ep}: reward={ep_reward:.1f}, steps={step+1}")

        return {
            'obs': np.array(all_obs, dtype=np.float32),
            'discrete_actions': np.array(all_discrete, dtype=np.int64),
            'cont_actions': np.array(all_cont, dtype=np.float32),
            'p2p_actions': np.array(all_p2p, dtype=np.float32),
            'rewards': np.array(all_rewards, dtype=np.float32),
            'safe_masks': np.array(all_safe_masks, dtype=np.float32),
        }

    def collect_all_scenarios(
            self,
            scenario_names: List[str] = None,
            n_episodes_per: int = 20,
            max_steps: int = 500,
            save_path: str = None,
    ) -> Dict[str, np.ndarray]:
        """Collect from multiple scenarios and merge."""
        if scenario_names is None:
            scenario_names = ['scenario_a', 'scenario_b', 'scenario_c']

        if save_path is None:
            save_path = os.path.join(tactical_dir, 'rl', 'expert_demos.pkl')

        merged = {
            'obs': [], 'discrete_actions': [], 'cont_actions': [],
            'p2p_actions': [], 'rewards': [], 'safe_masks': [],
        }

        for i, sc_name in enumerate(scenario_names):
            print(f"\n=== Collecting expert demos: {sc_name} ===")
            data = self.collect_scenario(
                sc_name,
                n_episodes=n_episodes_per,
                max_steps=max_steps,
                seed_base=i * 1000,
            )
            for k in merged:
                merged[k].append(data[k])

        # Concatenate
        result = {k: np.concatenate(v, axis=0) for k, v in merged.items()}
        n_total = len(result['obs'])
        print(f"\nTotal expert demos: {n_total} transitions")

        # Save
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"Saved to {save_path}")

        return result


if __name__ == '__main__':
    collector = ExpertDemoCollector()
    collector.collect_all_scenarios(
        scenario_names=['scenario_a', 'scenario_b'],
        n_episodes_per=15,
        max_steps=400,
    )

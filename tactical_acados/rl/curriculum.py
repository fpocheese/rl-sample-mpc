"""
Curriculum learning manager.

Implements phased training:
  Phase 1: Solo driving (no opponents) — learn track following
  Phase 2: Single slow opponent — learn basic overtaking
  Phase 3: Full scenarios with multiple opponents — learn tactical racing
  Phase 4: Randomized scenarios — generalization

Curriculum progresses based on success metrics (episode reward threshold).
"""

import os
import sys
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

dir_path = os.path.dirname(os.path.abspath(__file__))
tactical_dir = os.path.join(dir_path, '..')
sys.path.insert(0, tactical_dir)

from config import TacticalConfig


@dataclass
class CurriculumPhase:
    """Definition of a curriculum phase."""
    name: str
    scenario_name: str
    n_opponents: int
    speed_scale_range: tuple  # (min, max) for opponent speed scaling
    randomize_init: bool
    max_steps: int
    reward_threshold: float   # avg reward to advance
    min_episodes: int         # minimum episodes before advancing
    max_episodes: int         # force advance after this many


@dataclass
class CurriculumConfig:
    """Curriculum configuration."""
    phases: List[CurriculumPhase] = field(default_factory=lambda: [
        CurriculumPhase(
            name="Phase1_SoloDriving",
            scenario_name="scenario_a",
            n_opponents=0,
            speed_scale_range=(0.0, 0.0),
            randomize_init=False,
            max_steps=300,
            reward_threshold=50.0,
            min_episodes=30,
            max_episodes=80,
        ),
        CurriculumPhase(
            name="Phase2_SingleSlowOpp",
            scenario_name="scenario_a",
            n_opponents=1,
            speed_scale_range=(0.70, 0.80),
            randomize_init=True,
            max_steps=400,
            reward_threshold=40.0,
            min_episodes=50,
            max_episodes=120,
        ),
        CurriculumPhase(
            name="Phase3_FullScenario",
            scenario_name="scenario_a",
            n_opponents=2,
            speed_scale_range=(0.80, 0.90),
            randomize_init=True,
            max_steps=500,
            reward_threshold=30.0,
            min_episodes=80,
            max_episodes=200,
        ),
        CurriculumPhase(
            name="Phase4_MultiScenario",
            scenario_name="scenario_b",
            n_opponents=2,
            speed_scale_range=(0.82, 0.92),
            randomize_init=True,
            max_steps=500,
            reward_threshold=25.0,
            min_episodes=80,
            max_episodes=200,
        ),
    ])
    window_size: int = 20  # episodes for rolling avg


class CurriculumManager:
    """Manages curriculum progression during training."""

    def __init__(self, config: CurriculumConfig = None):
        self.config = config or CurriculumConfig()
        self.current_phase_idx = 0
        self.phase_episode_count = 0
        self.phase_rewards = []

    @property
    def current_phase(self) -> CurriculumPhase:
        idx = min(self.current_phase_idx, len(self.config.phases) - 1)
        return self.config.phases[idx]

    @property
    def is_complete(self) -> bool:
        return self.current_phase_idx >= len(self.config.phases)

    def should_advance(self) -> bool:
        """Check if we should advance to the next phase."""
        phase = self.current_phase
        if self.phase_episode_count < phase.min_episodes:
            return False
        if self.phase_episode_count >= phase.max_episodes:
            return True
        # Check reward threshold
        if len(self.phase_rewards) >= self.config.window_size:
            avg = np.mean(self.phase_rewards[-self.config.window_size:])
            if avg >= phase.reward_threshold:
                return True
        return False

    def advance(self):
        """Move to next phase."""
        old_name = self.current_phase.name
        self.current_phase_idx += 1
        self.phase_episode_count = 0
        self.phase_rewards = []
        if not self.is_complete:
            print(f"\n{'='*60}")
            print(f"CURRICULUM: {old_name} → {self.current_phase.name}")
            print(f"{'='*60}\n")

    def record_episode(self, reward: float):
        """Record episode result and check for advancement."""
        self.phase_episode_count += 1
        self.phase_rewards.append(reward)

        if self.should_advance() and not self.is_complete:
            self.advance()

    def get_env_kwargs(self) -> dict:
        """Get env configuration for current phase."""
        phase = self.current_phase
        return {
            'scenario_name': phase.scenario_name,
            'max_steps': phase.max_steps,
            'randomize_init': phase.randomize_init,
        }

    def modify_scenario_for_phase(self, scenario: dict) -> dict:
        """Modify loaded scenario to match current curriculum phase."""
        phase = self.current_phase
        modified = copy.deepcopy(scenario)

        # Limit opponents
        if 'opponents' in modified:
            modified['opponents'] = modified['opponents'][:phase.n_opponents]

        # Adjust speed scales
        for opp in modified.get('opponents', []):
            opp['speed_scale'] = np.random.uniform(
                phase.speed_scale_range[0],
                phase.speed_scale_range[1],
            )

        return modified

    def get_state(self) -> dict:
        """Serialize curriculum state for checkpointing."""
        return {
            'phase_idx': self.current_phase_idx,
            'phase_episode_count': self.phase_episode_count,
            'phase_rewards': self.phase_rewards,
        }

    def load_state(self, state: dict):
        """Restore curriculum state from checkpoint."""
        self.current_phase_idx = state['phase_idx']
        self.phase_episode_count = state['phase_episode_count']
        self.phase_rewards = state['phase_rewards']

    def summary(self) -> str:
        """Get curriculum summary string."""
        lines = []
        for i, phase in enumerate(self.config.phases):
            marker = "→" if i == self.current_phase_idx else " "
            done = "✓" if i < self.current_phase_idx else " "
            lines.append(
                f"  {marker}{done} Phase {i}: {phase.name} "
                f"(opp={phase.n_opponents}, thresh={phase.reward_threshold})"
            )
        return "\n".join(lines)

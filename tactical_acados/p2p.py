"""
Push-to-Pass (P2P) state machine.

- One activation per episode
- Duration = 15s
- +50hp equivalent → bounded longitudinal performance enhancement
"""

from config import TacticalConfig, DEFAULT_CONFIG


class PushToPass:
    """P2P state machine: one-shot activation with bounded duration."""

    def __init__(self, cfg: TacticalConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self.reset()

    def reset(self):
        """Reset P2P for new episode."""
        self._available = True
        self._active = False
        self._remaining_time = 0.0
        self._total_used_time = 0.0

    @property
    def available(self) -> bool:
        """True if P2P has not been used yet."""
        return self._available

    @property
    def active(self) -> bool:
        """True if P2P is currently active."""
        return self._active

    @property
    def remaining_time(self) -> float:
        """Remaining P2P time in seconds."""
        return self._remaining_time

    def activate(self) -> bool:
        """
        Try to activate P2P.
        Returns True if activation succeeded.
        """
        if not self._available:
            return False
        self._available = False
        self._active = True
        self._remaining_time = self.cfg.p2p_duration
        return True

    def step(self, dt: float):
        """Advance P2P state by dt seconds."""
        if not self._active:
            return

        self._remaining_time -= dt
        self._total_used_time += dt

        if self._remaining_time <= 0.0:
            self._active = False
            self._remaining_time = 0.0

    def get_speed_boost(self) -> float:
        """Get current speed boost (m/s). Zero if not active."""
        if self._active:
            return self.cfg.p2p_speed_boost
        return 0.0

    def get_gg_scale(self) -> float:
        """Get GG diagram scale factor for longitudinal acceleration."""
        if self._active:
            return 1.0 + self.cfg.p2p_power_boost_fraction
        return 1.0

    def get_state_vector(self):
        """State for observation: [available, active, remaining_frac]."""
        frac = self._remaining_time / self.cfg.p2p_duration if self.cfg.p2p_duration > 0 else 0.0
        return [float(self._available), float(self._active), frac]

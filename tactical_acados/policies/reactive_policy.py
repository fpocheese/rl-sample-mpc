"""
Reactive ACADOS baseline policy.

Treats opponents as pure obstacles to be carved out of the track boundaries. 
Lacks lookahead tactical reasoning or gap-dependent tracking behavior. 
Simply selects the overtake side greedily based on the ego vehicle's current lateral offset.
"""

from tactical_action import TacticalAction, DiscreteTactic, PreferenceVector
from observation import TacticalObservation
from config import TacticalConfig, DEFAULT_CONFIG

class ReactiveAcadosPolicy:
    """Reactive ACADOS baseline."""
    
    def __init__(self, cfg: TacticalConfig = DEFAULT_CONFIG):
        self.cfg = cfg

    def act(self, obs: TacticalObservation) -> TacticalAction:
        # Find closest opponent ahead (delta_s < 0 in observation frame)
        ahead = [o for o in obs.opponents if o.delta_s < 0 and abs(o.delta_s) < 150.0]
        
        if not ahead:
            # No opponents nearby, just run optimal raceline
            return TacticalAction(
                discrete_tactic=DiscreteTactic.RACE_LINE,
                aggressiveness=1.0,
                preference=PreferenceVector()
            )
            
        target = ahead[-1] # Closest opponent is the last one in the list among those ahead
        
        # Greedily dodge based on current lateral position
        if obs.ego_n > target.n:
            # We are to the left of opponent -> carve out the right side
            return TacticalAction(
                discrete_tactic=DiscreteTactic.OVERTAKE_LEFT,
                aggressiveness=1.0,  # Pure MPC tries to drive as fast as dynamically possible
                preference=PreferenceVector(rho_n=0.0) # No strategic lateral bias
            )
        else:
            # We are to the right of opponent -> carve out the left side
            return TacticalAction(
                discrete_tactic=DiscreteTactic.OVERTAKE_RIGHT,
                aggressiveness=1.0,
                preference=PreferenceVector(rho_n=0.0)
            )

#!/usr/bin/env python3
"""
Run sim_tactical visualisation from within the self-contained ROS2 package.

Usage (from *anywhere*):
    python  <path>/race24_tactical_planner_ros2/run_viz.py \
            --scenario scenario_c  [--policy heuristic] [--max-steps 99999]

The script mirrors exactly the original sim_tactical.py CLI but launches
from the package-internal copy, so every import and data path resolves
inside race24_tactical_planner_ros2/.
"""

import os
import sys

# ── 1. Resolve PACKAGE_ROOT ──────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = _THIS_DIR  # run_viz.py sits at package root

# ── 2. sys.path  (identical to tactical_planner_node.py) ─────────────
for _p in [
    _PACKAGE_ROOT,
    os.path.join(_PACKAGE_ROOT, 'src'),
    os.path.join(_PACKAGE_ROOT, 'tactical_acados'),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── 3. chdir so acados solver finds c_generated_code/ via relative
#       path in point_mass_ode_ocp.json ────────────────────────────────
os.chdir(_PACKAGE_ROOT)

# ── 4. Import the REAL sim_tactical that now lives inside the package ─
from sim_tactical import run_tactical_simulation           # noqa: E402

# ── 5. CLI (mirrors original sim_tactical __main__) ──────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Tactical Sim (self-contained ROS2 package)')
    parser.add_argument('--scenario', type=str, default='scenario_c',
                        help='scenario_a / scenario_b / scenario_c')
    parser.add_argument('--policy', type=str, default='heuristic',
                        choices=['heuristic', 'random', 'rl'])
    parser.add_argument('--max-steps', type=int, default=99999)
    parser.add_argument('--no-viz', action='store_true', help='Disable real-time plot')
    parser.add_argument('--force-side', type=str, default=None,
                        choices=['left', 'right'])
    parser.add_argument('--follow-when-forced', action='store_true', default=True)
    parser.add_argument('--no-follow-when-forced', action='store_true')
    args = parser.parse_args()

    _follow = not args.no_follow_when_forced

    run_tactical_simulation(
        scenario_name=args.scenario,
        max_steps=args.max_steps,
        visualize=not args.no_viz,
        policy_type=args.policy,
        force_side=args.force_side,
        follow_when_forced=_follow,
    )

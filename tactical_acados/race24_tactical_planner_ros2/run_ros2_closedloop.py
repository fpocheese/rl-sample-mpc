#!/usr/bin/env python3
"""
ROS2 closed-loop test — single process launcher.

Starts sim_env_node + tactical_planner_node in the same process with a
MultiThreadedExecutor, avoiding colcon-build ceremony.  Perfect for fast
iteration during development.

When visualisation is enabled (default), the ROS2 executor runs on a
background thread while the main thread pumps the Matplotlib GUI event
loop.  This avoids the "main thread is not in main loop" Tk error.

Usage:
  source /opt/ros/humble/setup.bash
  source /home/uav/race24/Racecar/install/setup.bash
  python3 run_ros2_closedloop.py [--scenario scenario_c] [--max-steps 99999] [--no-viz]
"""

import os
import sys
import time
import argparse
import threading

# ── Path setup ───────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = _THIS_DIR
for _p in [_PACKAGE_ROOT,
           os.path.join(_PACKAGE_ROOT, 'src'),
           os.path.join(_PACKAGE_ROOT, 'tactical_acados')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_PACKAGE_ROOT)

import rclpy
from rclpy.executors import MultiThreadedExecutor

from tactical_planner_ros2.tactical_planner_node import TacticalPlannerNode
from tactical_planner_ros2.sim_env_node import SimEnvNode


# ── Helpers ──────────────────────────────────────────────────────────

def _spin_until_done(executor, sim_node):
    """Spin the ROS2 executor until the sim signals _finished.

    Uses spin() (blocking) which is more efficient for a
    MultiThreadedExecutor than a spin_once loop.  The daemon flag on the
    thread ensures we exit when the main thread finishes.
    """
    try:
        executor.spin()
    except Exception:
        pass


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ROS2 closed-loop sim')
    parser.add_argument('--scenario', default='scenario_c')
    parser.add_argument('--policy', default='heuristic')
    parser.add_argument('--max-steps', type=int, default=99999)
    parser.add_argument('--no-viz', action='store_true')
    parser.add_argument('--force-side', default='none')
    args = parser.parse_args()

    do_viz = not args.no_viz

    ros_args = [
        '--ros-args',
        '-p', f'scenario:={args.scenario}',
        '-p', f'max_steps:={args.max_steps}',
        '-p', f'visualize:={do_viz}',
        '-p', f'policy_type:={args.policy}',
        '-p', f'force_side:={args.force_side}',
    ]

    rclpy.init(args=ros_args)

    sim_node = SimEnvNode()
    planner_node = TacticalPlannerNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(sim_node)
    executor.add_node(planner_node)

    print('=' * 60)
    print('ROS2 Closed-Loop: sim_env_node ↔ tactical_planner_node')
    print(f'  Scenario : {args.scenario}')
    print(f'  Policy   : {args.policy}')
    print(f'  Max steps: {args.max_steps}')
    print(f'  Viz      : {do_viz}')
    print('=' * 60, flush=True)

    if do_viz:
        # ── Matplotlib must run on the main thread ──────────────────
        # Spin the ROS2 executor on a daemon thread; the main thread
        # handles the GUI event loop (Qt/Tk) via pump_viz + plt.pause.
        import matplotlib.pyplot as plt

        spin_thread = threading.Thread(
            target=_spin_until_done,
            args=(executor, sim_node),
            daemon=True,
        )
        spin_thread.start()

        # Pump GUI event loop until the sim is done
        try:
            while not getattr(sim_node, '_finished', False):
                # Drain viz queue (calls _viz.update on main thread)
                sim_node.pump_viz()
                # Let the GUI backend process pending events (~20 fps)
                plt.pause(0.05)
        except KeyboardInterrupt:
            pass

        # Stop the executor so planner & sim nodes stop spinning
        executor.shutdown()
        spin_thread.join(timeout=2.0)

        # Drain any remaining viz data
        while sim_node.pump_viz():
            pass

        # Keep the final plot open so the user can inspect it
        if sim_node._viz is not None:
            print('[viz] Simulation done. Close the plot window to exit.')
            plt.ioff()
            plt.show()

    else:
        # ── No viz: spin in the main thread ─────────────────────────
        try:
            while rclpy.ok():
                executor.spin_once(timeout_sec=0.1)
                if getattr(sim_node, '_finished', False):
                    break
        except (KeyboardInterrupt, SystemExit):
            pass

    # ── Cleanup ──────────────────────────────────────────────────────
    sim_node.destroy_node()
    planner_node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()

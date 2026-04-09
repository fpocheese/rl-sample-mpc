#!/usr/bin/env python3
"""
ROS2 Simulation Environment Node  –  **Synchronous event-driven** variant
===========================================================================

Replaces the real car + observer + V2V system with a perfect-tracking
simulation loop.  Publishes exactly the same ROS2 topics that the real
race24 system does, so tactical_planner_node sees *identical* messages.

Architecture (event-driven, NOT timer-polled)
─────────────────────────────────────────────
  1.  On startup → publish initial ego / opponent data → planner sees it
  2.  Planner plans → publishes trajectory + status
  3.  On trajectory callback:
        a.  Decode trajectory (using Frenet s,n,chi,t directly from msg)
        b.  Enqueue viz data (with corridor bounds from status)
        c.  perfect_tracking_update  (advance ego by assumed_calc_time)
        d.  Step opponents
        e.  Collision check  /  log  /  boundary check
        f.  Re-publish ego + opponents  → triggers next planner cycle

This eliminates the 8 Hz timer that could "miss" trajectory arrivals,
making the sim deterministic and as fast as the planner can go.

Publish  (same topics as the real observer / V2V stack)
────────
  /flyeagle/a2rl/observer/ego_loc       Localization
  /flyeagle/a2rl/observer/ego_state     EgoState
  flyeagle/v2v_ground_truth             GroundTruthArray

Subscribe  (output of tactical_planner_node)
─────────
  /flyeagle/a2rl/tactical_planner/trajectory   ReferencePath
  /tactical_planner/status                      String   (JSON)
"""

import os
import sys
import time
import math
import json
import yaml
import collections
import numpy as np
import threading
from typing import Optional

# ── path setup (same as tactical_planner_node) ──────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
for _p in [_PACKAGE_ROOT,
           os.path.join(_PACKAGE_ROOT, 'src'),
           os.path.join(_PACKAGE_ROOT, 'tactical_acados')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_PACKAGE_ROOT)

# ── ROS2 ─────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from std_msgs.msg import String
from a2rl_bs_msgs.msg import (
    Localization,
    EgoState,
    ReferencePath,
    CartesianFrame,
    Timestamp,
)
from autonoma_msgs.msg import GroundTruthArray, GroundTruth

# ── Algorithm imports (track, opponents, perfect tracking) ───────────
from tactical_acados.config import TacticalConfig
from tactical_acados.sim_acados_only import (
    load_setup, create_initial_state, perfect_tracking_update,
)
from tactical_acados.opponent import OpponentVehicle


# =====================================================================
# Helper
# =====================================================================
def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _ns_now():
    return int(time.time() * 1e9)


# Lightweight container so visualizer_tactical can read .n_left_override etc.
class _GuidanceProxy:
    """Minimal guidance-like object for the visualizer corridor lines."""
    __slots__ = ('n_left_override', 'n_right_override')

    def __init__(self, n_left=None, n_right=None):
        self.n_left_override = np.asarray(n_left, dtype=float) \
            if n_left is not None else None
        self.n_right_override = np.asarray(n_right, dtype=float) \
            if n_right is not None else None


# =====================================================================
# Node
# =====================================================================
class SimEnvNode(Node):
    """ROS2 simulation environment that mimics the real car's observers."""

    def __init__(self):
        super().__init__('sim_env_node')

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter('scenario', 'scenario_c')
        self.declare_parameter('max_steps', 99999)
        self.declare_parameter('timer_hz', 8.0)
        self.declare_parameter('track_name', 'yas_user_smoothed')
        self.declare_parameter('vehicle_name', 'eav25_car')
        self.declare_parameter('raceline_name',
                               'yasnorth_3d_rl_as_ref_eav25_car_gg_0.1')
        self.declare_parameter('visualize', True)

        scenario_name = self.get_parameter('scenario').value
        self._max_steps = self.get_parameter('max_steps').value
        track_name = self.get_parameter('track_name').value
        vehicle_name = self.get_parameter('vehicle_name').value
        raceline_name = self.get_parameter('raceline_name').value
        self._do_viz = self.get_parameter('visualize').value

        # ── Load scenario YAML ───────────────────────────────────────
        scenario_dir = os.path.join(_PACKAGE_ROOT, 'tactical_acados',
                                    'scenarios')
        with open(os.path.join(scenario_dir, f'{scenario_name}.yml')) as f:
            scenario = yaml.safe_load(f)

        sc = scenario['scenario']
        ego_cfg = scenario['ego']
        opp_cfgs = scenario.get('opponents', [])
        planner_cfg = scenario.get('planner', {})

        # ── Core setup ───────────────────────────────────────────────
        self.cfg = TacticalConfig()
        self.cfg.optimization_horizon_m = planner_cfg.get(
            'optimization_horizon_m', 500.0)
        self.cfg.gg_margin = planner_cfg.get('gg_margin', 0.1)
        self.cfg.safety_distance_default = planner_cfg.get(
            'safety_distance', 0.5)
        self.cfg.assumed_calc_time = planner_cfg.get(
            'assumed_calc_time', 0.125)

        self.params, self.track_handler, self.gg_handler, \
            self._local_planner, self.global_planner = load_setup(
                self.cfg,
                track_name=track_name,
                vehicle_name=vehicle_name,
                raceline_name=raceline_name,
                init_local_planner=False,
            )

        # ── Initial ego state ────────────────────────────────────────
        self.ego = create_initial_state(
            self.track_handler,
            start_s=ego_cfg['start_s'],
            start_n=ego_cfg['start_n'],
            start_V=ego_cfg['start_V'],
            start_chi=ego_cfg.get('start_chi', 0.0),
            start_ax=ego_cfg.get('start_ax', 0.0),
            start_ay=ego_cfg.get('start_ay', 0.0),
        )

        # ── Opponents ────────────────────────────────────────────────
        self.opponents = []
        for opp_cfg in opp_cfgs:
            opp = OpponentVehicle(
                vehicle_id=opp_cfg['id'],
                s_init=opp_cfg['start_s'],
                n_init=opp_cfg.get('start_n', 0.0),
                V_init=opp_cfg.get('start_V', 40.0),
                track_handler=self.track_handler,
                global_planner=self.global_planner,
                speed_scale=opp_cfg.get('speed_scale', 0.85),
                cfg=self.cfg,
            )
            self.opponents.append(opp)

        # ── Scenario boundary ────────────────────────────────────────
        self._s_end = sc.get('s_end', 1e6)
        self._sc_name = sc['name']

        # ── Runtime ──────────────────────────────────────────────────
        self._step = 0
        self._collision_count = 0
        self._sim_lock = threading.Lock()   # serialise sim step
        self._finished = False              # set True by _finish()

        # Logging for comparison
        self._log_s = []
        self._log_n = []
        self._log_V = []
        self._log_tactic = []

        # Latest corridor bounds (from status JSON) for viz
        self._corridor_left = None   # np.ndarray or None
        self._corridor_right = None

        # ── QoS ──────────────────────────────────────────────────────
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST, depth=1)

        # ── Publishers (mimic real car) ──────────────────────────────
        self.pub_loc = self.create_publisher(
            Localization,
            '/flyeagle/a2rl/observer/ego_loc', qos_sensor)

        self.pub_ego = self.create_publisher(
            EgoState,
            '/flyeagle/a2rl/observer/ego_state', qos_sensor)

        self.pub_v2v = self.create_publisher(
            GroundTruthArray,
            'flyeagle/v2v_ground_truth', qos_sensor)

        # ── Callback Groups ───────────────────────────────────────
        # Separate groups so subscriber callbacks can run in parallel
        # with the timer callback in a MultiThreadedExecutor.
        self._sub_cb_group = MutuallyExclusiveCallbackGroup()
        self._status_cb_group = MutuallyExclusiveCallbackGroup()
        self._timer_cb_group = MutuallyExclusiveCallbackGroup()

        # ── Subscribers (from tactical_planner_node) ─────────────────
        self.sub_traj = self.create_subscription(
            ReferencePath,
            '/flyeagle/a2rl/tactical_planner/trajectory',
            self._cb_trajectory, qos_reliable,
            callback_group=self._sub_cb_group)

        # Status in its own group so corridor data arrives promptly
        # even while _cb_trajectory is running _sim_step.
        self.sub_status = self.create_subscription(
            String,
            '/tactical_planner/status',
            self._cb_status, qos_reliable,
            callback_group=self._status_cb_group)

        self._latest_status = {}

        # ── Visualization ────────────────────────────────────────────
        self._viz = None
        self._viz_queue = collections.deque(maxlen=1)   # main-thread viz
        if self._do_viz:
            try:
                from tactical_acados.visualizer_tactical import (
                    TacticalVisualizer,
                )
                self._viz = TacticalVisualizer(
                    self.track_handler, self.gg_handler, self.params,
                    n_opponents=len(self.opponents),
                    global_planner=self.global_planner,
                )
            except Exception as e:
                self.get_logger().warn(f'Viz init failed: {e}')

        # ── Kick-start: publish initial ego / opponent data ──────────
        # Use a one-shot timer so the executor has time to register subs.
        self._kickstart_timer = self.create_timer(
            0.3, self._kickstart_cb,
            callback_group=self._timer_cb_group)

        self.get_logger().info(
            f'\033[1;36m[SimEnv] started  scenario={scenario_name}  '
            f'opponents={len(self.opponents)}\033[0m')

    # ==================================================================
    #  Kick-start (one-shot)
    # ==================================================================

    def _kickstart_cb(self):
        """Publish initial ego data so the planner has something to
        plan on.  Then cancel this timer.
        """
        self._kickstart_timer.cancel()
        self.get_logger().info(
            '[SimEnv] Publishing initial ego/opp data to trigger planner...')
        self._publish_localization()
        self._publish_ego_state()
        self._publish_opponents()

    # ==================================================================
    #  Callbacks
    # ==================================================================

    def _cb_trajectory(self, msg: ReferencePath):
        """Decode ReferencePath and immediately execute one simulation step.

        The Frenet state (s, n, chi, t) is read directly from spare message
        fields written by tactical_planner_node, avoiding lossy cartesian→
        Frenet back-conversion.

        Encoding (see tactical_planner_node._publish_trajectory):
          orientation_ypr.z  = s
          velocity_angular.x = n
          velocity_angular.y = chi
          acceleration.z     = t
        """
        if self._finished:
            return

        # ── Decode trajectory ────────────────────────────────────────
        traj = {'s': [], 'n': [], 'V': [], 'chi': [],
                'x': [], 'y': [], 'z': [], 't': [],
                'ax': [], 'ay': []}

        dt_fallback = float(msg.path_time_discretization_s) \
            if msg.path_time_discretization_s > 0 else 0.125

        for i, pt in enumerate(msg.path):
            x_g = float(pt.orientation_ypr.x)
            y_g = float(pt.orientation_ypr.y)
            s_i = float(pt.orientation_ypr.z)      # Frenet s (direct)
            speed = float(pt.velocity_linear.x)
            n_i = float(pt.velocity_angular.x)     # Frenet n (direct)
            chi_i = float(pt.velocity_angular.y)   # Frenet chi (direct)
            t_i = float(pt.acceleration.z)          # time (direct)

            # Fallback: if s is 0 for all points past index 0, the
            # publisher may be an older version → fall back to cartesian2sn.
            if s_i == 0.0 and i > 0:
                try:
                    sn = self.track_handler.cartesian2sn(x_g, y_g)
                    s_i = float(sn[0])
                    n_i = float(sn[1])
                except Exception:
                    s_i = traj['s'][-1] if traj['s'] else 0.0
                    n_i = traj['n'][-1] if traj['n'] else 0.0
                yaw = float(pt.velocity_linear.y)
                road_yaw = float(np.interp(
                    s_i, self.track_handler.s, self.track_handler.psi,
                    period=self.track_handler.s[-1]))
                chi_i = _wrap(yaw - road_yaw)
                t_i = i * dt_fallback

            traj['s'].append(s_i)
            traj['n'].append(n_i)
            traj['V'].append(speed)
            traj['chi'].append(chi_i)
            traj['x'].append(x_g)
            traj['y'].append(y_g)
            traj['z'].append(0.0)
            traj['t'].append(t_i)
            traj['ax'].append(float(pt.acceleration.x))
            traj['ay'].append(float(pt.acceleration.y))

        # Convert lists to numpy arrays
        for k in traj:
            traj[k] = np.array(traj[k])

        if len(traj['s']) == 0:
            return

        # ── Execute one simulation step (synchronous) ────────────────
        with self._sim_lock:
            self._sim_step(traj)

    def _cb_status(self, msg: String):
        """Decode status JSON — may include corridor bounds."""
        try:
            data = json.loads(msg.data)
            self._latest_status = data
            # Extract corridor bounds for viz
            nl = data.get('n_left_override')
            nr = data.get('n_right_override')
            if nl is not None:
                self._corridor_left = np.array(nl, dtype=float)
            else:
                self._corridor_left = None
            if nr is not None:
                self._corridor_right = np.array(nr, dtype=float)
            else:
                self._corridor_right = None
        except Exception:
            pass

    # ==================================================================
    #  Simulation step
    # ==================================================================

    def _sim_step(self, traj: dict):
        """Execute one complete simulation step: viz → track → step → pub.

        Called inside _sim_lock from the trajectory subscriber callback.
        """
        if self._step >= self._max_steps:
            self._finish()
            return

        # ── Viz enqueue (BEFORE updating ego, so we show the state
        #    that produced this trajectory) ───────────────────────────
        if self._viz is not None:
            tact = self._latest_status.get('tactic', '?')
            carver = self._latest_status.get('carver_mode', '?')
            c_side = self._latest_status.get('carver_side', '')
            ot_rdy = ' OT_RDY!' if self._latest_status.get(
                'overtake_ready', False) else ''

            tactical_info = (
                f"Mode:  {tact}\n"
                f"Carver: {carver} {c_side or ''}{ot_rdy}\n"
                f"Step: {self._step}"
            )

            opp_preds = [opp.predict() for opp in self.opponents]
            for opp, pred in zip(self.opponents, opp_preds):
                st = opp.get_state()
                pred['x'] = st['x']
                pred['y'] = st['y']
                pred['s'] = st['s']
                pred['chi'] = st.get('chi', 0.0)

            # Build guidance proxy for corridor lines
            guidance_proxy = None
            if self._corridor_left is not None:
                guidance_proxy = _GuidanceProxy(
                    self._corridor_left, self._corridor_right)

            self._viz_queue.append(dict(
                ego=dict(self.ego),
                traj={k: np.array(v) for k, v in traj.items()},
                opponents=opp_preds,
                tactical_info=tactical_info,
                guidance=guidance_proxy,
            ))

        # ── Perfect tracking update ──────────────────────────────────
        self.ego = perfect_tracking_update(
            self.ego, traj,
            self.cfg.assumed_calc_time,
            self.track_handler,
        )

        # ── Step opponents ───────────────────────────────────────────
        for opp in self.opponents:
            opp.step(self.cfg.assumed_calc_time, self.ego)

        # ── Collision check ──────────────────────────────────────────
        for opp in self.opponents:
            dist = math.sqrt(
                (self.ego['x'] - opp.x) ** 2 +
                (self.ego['y'] - opp.y) ** 2)
            if dist < self.cfg.vehicle_length * 0.5:
                self._collision_count += 1
                self.get_logger().warn(
                    f'*** COLLISION step={self._step} '
                    f'Opp{opp.vehicle_id} dist={dist:.2f} ***')

        # ── Log ──────────────────────────────────────────────────────
        self._log_s.append(self.ego['s'])
        self._log_n.append(self.ego['n'])
        self._log_V.append(self.ego['V'])
        self._log_tactic.append(
            self._latest_status.get('tactic', '?'))

        # ── Periodic print ───────────────────────────────────────────
        if self._step % 20 == 0:
            tact = self._latest_status.get('tactic', '?')
            carver = self._latest_status.get('carver_mode', '?')
            c_side = self._latest_status.get('carver_side', '') or ''
            plan_ms = self._latest_status.get('plan_ms', 0)
            opp_info = ' | '.join(
                [f'Opp{o.vehicle_id}: s={o.s:.0f} n={o.n:.1f} '
                 f'{o.tactic}'
                 for o in self.opponents])
            self.get_logger().info(
                f'[{self._step:4d}] s={self.ego["s"]:7.1f} '
                f'n={self.ego["n"]:5.2f} V={self.ego["V"]:5.1f} | '
                f'{tact:10s} | {carver:8s} {c_side:5s} | '
                f'{opp_info} | {plan_ms:.0f}ms')

        # ── Scenario boundary ────────────────────────────────────────
        if self.ego['s'] > self._s_end:
            self.get_logger().info(
                f'*** Scenario boundary s_end={self._s_end} reached ***')
            self._finish()
            return

        self._step += 1

        # ── Re-publish ego / opponents → planner reads on next tick ──
        self._publish_localization()
        self._publish_ego_state()
        self._publish_opponents()

    # ==================================================================
    #  Publishers
    # ==================================================================

    def _publish_localization(self):
        """Publish a2rl_bs_msgs/Localization from ego_state dict."""
        msg = Localization()
        ns = _ns_now()
        msg.timestamp.nanoseconds = ns

        msg.position.x = float(self.ego['x'])
        msg.position.y = float(self.ego['y'])
        msg.position.z = float(self.ego.get('z', 0.0))

        # orientation_ypr.z = yaw (global heading)
        s = self.ego['s']
        road_yaw = float(np.interp(
            s, self.track_handler.s, self.track_handler.psi,
            period=self.track_handler.s[-1]))
        yaw = road_yaw + self.ego['chi']
        msg.orientation_ypr.z = float(yaw)

        self.pub_loc.publish(msg)

    def _publish_ego_state(self):
        """Publish a2rl_bs_msgs/EgoState from ego_state dict."""
        msg = EgoState()
        msg.timestamp.nanoseconds = _ns_now()

        V = self.ego['V']
        chi = self.ego['chi']
        s = self.ego['s']

        # velocity in ego-local frame (vx = forward, vy = lateral)
        # In sim: body-frame velocity = (V*cos(chi), V*sin(chi))
        # But in the real system, EgoState.velocity.x/y are in *global* NED
        # Actually checking tactical_planner_node._build_ego_state:
        #   vx = eg.velocity.x,  vy = eg.velocity.y
        #   V = sqrt(vx^2 + vy^2)
        # So we need global velocity components.
        road_yaw = float(np.interp(
            s, self.track_handler.s, self.track_handler.psi,
            period=self.track_handler.s[-1]))
        yaw = road_yaw + chi
        msg.velocity.x = float(V * math.cos(yaw))
        msg.velocity.y = float(V * math.sin(yaw))
        msg.velocity.z = 0.0

        msg.acceleration.x = float(self.ego.get('ax', 0.0))
        msg.acceleration.y = float(self.ego.get('ay', 0.0))

        self.pub_ego.publish(msg)

    def _publish_opponents(self):
        """Publish autonoma_msgs/GroundTruthArray.

        Real V2V message semantics:
          del_x = forward distance in ego body frame [m] (positive = ahead)
          del_y = rightward distance in ego body frame [m] (positive = right)
          vx    = *relative* longitudinal velocity in ego body frame [m/s]
                  (opp_vx_body_proj - ego_vx_body)
          vy    = *relative* lateral velocity in ego body frame [m/s]
                  (opp_vy_body_proj - ego_vy_body)
          yaw   = *relative* heading: wrap(ego_yaw - opp_yaw) [rad]
          car_num = opponent ID
        """
        arr = GroundTruthArray()

        ego_x = self.ego['x']
        ego_y = self.ego['y']
        s = self.ego['s']
        road_yaw = float(np.interp(
            s, self.track_handler.s, self.track_handler.psi,
            period=self.track_handler.s[-1]))
        ego_yaw = road_yaw + self.ego['chi']
        ego_V = self.ego['V']
        ego_chi = self.ego['chi']

        cos_y = math.cos(ego_yaw)
        sin_y = math.sin(ego_yaw)

        # Ego velocity in body frame
        ego_vx_body = ego_V * math.cos(ego_chi)  # forward component
        ego_vy_body = ego_V * math.sin(ego_chi)   # lateral component (≈0 small chi)

        for opp in self.opponents:
            gt = GroundTruth()
            gt.car_num = opp.vehicle_id

            # ---- Position: global → ego body-local ----
            dx_global = opp.x - ego_x
            dy_global = opp.y - ego_y
            del_x_local = cos_y * dx_global + sin_y * dy_global   # forward
            del_y_local = sin_y * dx_global - cos_y * dy_global   # rightward

            gt.del_x = float(del_x_local)     # forward  (positive = ahead)
            gt.del_y = float(del_y_local)      # rightward (positive = right)

            # ---- Velocity: compute opponent global vel, project to ego body,
            #      subtract ego body vel → relative velocity ----
            opp_road_yaw = float(np.interp(
                opp.s, self.track_handler.s, self.track_handler.psi,
                period=self.track_handler.s[-1]))
            opp_yaw_global = opp_road_yaw + opp.chi

            opp_vx_global = opp.V * math.cos(opp_yaw_global)
            opp_vy_global = opp.V * math.sin(opp_yaw_global)

            # Project opponent global velocity into ego body frame
            opp_vx_in_ego_body = cos_y * opp_vx_global + sin_y * opp_vy_global
            opp_vy_in_ego_body = sin_y * opp_vx_global - cos_y * opp_vy_global

            # Publish absolute opponent global velocity components (m/s)
            # Previous versions of the tactical stack expected absolute
            # velocity components; publish those to maintain compatibility.
            gt.vx = float(opp_vx_global)
            gt.vy = float(opp_vy_global)
            gt.vz = 0.0

            # ---- Heading: relative yaw = wrap(ego_yaw - opp_yaw) ----
            gt.yaw = float(_wrap(ego_yaw - opp_yaw_global))

            gt.pitch = 0.0
            gt.roll = 0.0
            gt.del_z = 0.0

            arr.vehicles.append(gt)

        self.pub_v2v.publish(arr)

    # ==================================================================
    #  Finish
    # ==================================================================

    def _finish(self):
        n = len(self._log_s)
        if n == 0:
            self.get_logger().info('No steps completed.')
            self._finished = True
            return

        self.get_logger().info('=' * 60)
        self.get_logger().info(
            f'Simulation complete: {n} steps, '
            f'{self._collision_count} collisions')
        self.get_logger().info(
            f'  s range: {self._log_s[0]:.1f} → {self._log_s[-1]:.1f}')
        self.get_logger().info(
            f'  V mean={np.mean(self._log_V):.1f} '
            f'max={np.max(self._log_V):.1f} m/s')
        self.get_logger().info(
            f'  n mean={np.mean(np.abs(self._log_n)):.3f} '
            f'max={np.max(np.abs(self._log_n)):.3f} m')
        self.get_logger().info('=' * 60)

        self._finished = True

    # ==================================================================
    #  pump_viz  — called from the MAIN thread
    # ==================================================================

    def pump_viz(self):
        """Drain the viz queue and call _viz.update() on the main thread.

        Returns True if an update was drawn, False otherwise.
        """
        if self._viz is None:
            return False
        try:
            data = self._viz_queue.popleft()
        except IndexError:
            return False
        self._viz.update(
            data['ego'], data['traj'],
            opponents=data['opponents'],
            tactical_info=data['tactical_info'],
            guidance=data.get('guidance'),
        )
        return True


def main(args=None):
    rclpy.init(args=args)
    node = SimEnvNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

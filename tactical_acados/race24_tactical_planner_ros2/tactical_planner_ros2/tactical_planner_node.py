"""
Tactical Planner ROS2 Node
===========================

Wraps the tactical (heuristic / RL) planner into a ROS2 lifecycle-aware
node that integrates with the A2RL planner_cvxopt framework.

Subscriptions
-------------
- /flyeagle/a2rl/observer/ego_loc      (a2rl_bs_msgs/Localization)
- /flyeagle/a2rl/observer/ego_state    (a2rl_bs_msgs/EgoState)
- flyeagle/v2v_ground_truth            (autonoma_msgs/GroundTruthArray)
- /tactical_planner/param/force_side    (std_msgs/String)
- /tactical_planner/param/follow_when_forced (std_msgs/Bool)

Publications
------------
- /flyeagle/a2rl/tactical_planner/trajectory  (a2rl_bs_msgs/ReferencePath)
- /tactical_planner/status                     (std_msgs/String)        JSON status
- /tactical_planner/fsm_state                  (std_msgs/String)        JSON FSM details
- /tactical_planner/feasible_domain            (visualization_msgs/MarkerArray) Foxglove

Corresponding to case 16 in planner.cpp (sel_track_mode=13, local_planner_method_=6).
"""

import os
import sys
import time
import math
import json
import traceback
import threading
from typing import Optional

import numpy as np

# ------------------------------------------------------------------
# Path setup: self-contained ROS2 package layout
#   race24_tactical_planner_ros2/   ← _PACKAGE_ROOT
#   ├── tactical_planner_ros2/      ← _THIS_DIR (this node)
#   ├── tactical_acados/            ← algorithm library
#   ├── src/                        ← track3D, ggManager, etc.
#   ├── data/                       ← track/gg/vehicle/raceline data
#   └── c_generated_code/           ← acados .so
# ------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
for _p in [_PACKAGE_ROOT, os.path.join(_PACKAGE_ROOT, 'src'), os.path.join(_PACKAGE_ROOT, 'tactical_acados')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Ensure acados solver can locate c_generated_code/ and point_mass_ode_ocp.json
os.chdir(_PACKAGE_ROOT)

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

from std_msgs.msg import String, Bool, Float64MultiArray, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

# a2rl / autonoma message types
from a2rl_bs_msgs.msg import (
    Localization,
    EgoState,
    ReferencePath,
    CartesianFrameState,
    CartesianFrame,
    Timestamp,
)
from autonoma_msgs.msg import GroundTruthArray

# Tactical planner core imports
from tactical_acados.config import TacticalConfig
from tactical_acados.acados_planner import AcadosTacticalPlanner
from tactical_acados.tactical_action import (
    TacticalAction, PlannerGuidance, get_fallback_action, TacticalMode,
)
from tactical_acados.observation import TacticalObservation, build_observation
from tactical_acados.safe_wrapper import SafeTacticalWrapper
from tactical_acados.planner_guidance import TacticalToPlanner
from tactical_acados.opponent import OpponentVehicle
from tactical_acados.p2p import PushToPass
from tactical_acados.follow_module import FollowModule
from tactical_acados.a2rl_obstacle_carver import A2RLObstacleCarver, CarverMode
from tactical_acados.sim_acados_only import load_setup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_PATH_POINTS = 30
_CARVER_MODE_MAP = {
    'follow': CarverMode.FOLLOW,
    'shadow': CarverMode.SHADOW,
    'overtake': CarverMode.OVERTAKE,
    'raceline': CarverMode.RACELINE,
    'hold': CarverMode.HOLD,
    'force_left': CarverMode.FORCE_LEFT,
    'force_right': CarverMode.FORCE_RIGHT,
}


class TacticalPlannerNode(Node):
    """ROS2 node wrapping the tactical planner pipeline."""

    def __init__(self):
        super().__init__('tactical_planner_node')

        # ── ROS2 Parameters ──────────────────────────────────────────
        self.declare_parameter('policy_type', 'heuristic')
        self.declare_parameter('force_side', 'none')
        self.declare_parameter('follow_when_forced', True)
        self.declare_parameter('timer_hz', 20.0)
        self.declare_parameter('scenario', 'scenario_c')
        self.declare_parameter('track_name', 'yas_user_smoothed')
        self.declare_parameter('vehicle_name', 'eav25_car')
        self.declare_parameter('raceline_name',
                               'yasnorth_3d_rl_as_ref_eav25_car_gg_0.1')

        policy_type = self.get_parameter('policy_type').value
        force_side_str = self.get_parameter('force_side').value
        follow_when_forced = self.get_parameter('follow_when_forced').value
        timer_hz = self.get_parameter('timer_hz').value
        track_name = self.get_parameter('track_name').value
        vehicle_name = self.get_parameter('vehicle_name').value
        raceline_name = self.get_parameter('raceline_name').value

        force_side = None if force_side_str in ('none', '') else force_side_str

        # ── Core planner setup (reuses sim_tactical / sim_acados_only) ──
        self.cfg = TacticalConfig()
        self.params, self.track_handler, self.gg_handler, \
            self.local_planner, self.global_planner = load_setup(
                self.cfg,
                track_name=track_name,
                vehicle_name=vehicle_name,
                raceline_name=raceline_name,
            )

        self.planner = AcadosTacticalPlanner(
            local_planner=self.local_planner,
            global_planner=self.global_planner,
            track_handler=self.track_handler,
            vehicle_params=self.params['vehicle_params'],
            cfg=self.cfg,
        )

        self.safe_wrapper = SafeTacticalWrapper(self.cfg)
        self.tactical_mapper = TacticalToPlanner(self.track_handler, self.cfg)
        self.p2p = PushToPass(self.cfg)
        self.a2rl_carver = A2RLObstacleCarver(
            self.track_handler, self.cfg,
            global_planner=self.global_planner,
        )
        self.follow_mod = FollowModule(self.track_handler, self.cfg)

        # ── Policy ───────────────────────────────────────────────────
        if policy_type == 'heuristic':
            from tactical_acados.policies.heuristic_policy import (
                HeuristicTacticalPolicy,
            )
            self.policy = HeuristicTacticalPolicy(
                self.cfg,
                force_side=force_side,
                follow_when_forced=follow_when_forced,
            )
        elif policy_type == 'rl':
            from tactical_acados.policies.rl_policy import RLTacticalPolicy
            ckpt = os.path.join(_PKG_DIR, 'checkpoints', 'best_policy.pt')
            self.policy = RLTacticalPolicy(ckpt, self.cfg)
        else:
            from tactical_acados.policies.heuristic_policy import (
                HeuristicTacticalPolicy,
            )
            self.policy = HeuristicTacticalPolicy(self.cfg)

        self.prev_action = get_fallback_action()

        # ── Runtime state ────────────────────────────────────────────
        self._ego_state: Optional[dict] = None
        self._loc_msg: Optional[Localization] = None
        self._ego_msg: Optional[EgoState] = None
        self._opp_detections: list = []          # list of GroundTruth
        self._loc_stamp = self.get_clock().now()
        self._ego_stamp = self.get_clock().now()
        self._opp_stamp = self.get_clock().now()
        self._step = 0
        self._data_lock = threading.Lock()       # protect shared msg data

        # ── QoS ──────────────────────────────────────────────────────
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Callback Groups ─────────────────────────────────────────
        # Separate groups so subscriber callbacks can run in parallel
        # with the timer callback in a MultiThreadedExecutor.
        self._sub_cb_group = MutuallyExclusiveCallbackGroup()
        self._timer_cb_group = MutuallyExclusiveCallbackGroup()

        # ── Subscribers ──────────────────────────────────────────────
        self.sub_loc = self.create_subscription(
            Localization,
            '/flyeagle/a2rl/observer/ego_loc',
            self._cb_localization, qos_sensor,
            callback_group=self._sub_cb_group)

        self.sub_ego = self.create_subscription(
            EgoState,
            '/flyeagle/a2rl/observer/ego_state',
            self._cb_ego_state, qos_sensor,
            callback_group=self._sub_cb_group)

        self.sub_opp = self.create_subscription(
            GroundTruthArray,
            'flyeagle/v2v_ground_truth',
            self._cb_opponents, qos_sensor,
            callback_group=self._sub_cb_group)

        # Runtime parameter topics
        self.sub_force_side = self.create_subscription(
            String,
            '/tactical_planner/param/force_side',
            self._cb_force_side, qos_reliable,
            callback_group=self._sub_cb_group)

        self.sub_follow = self.create_subscription(
            Bool,
            '/tactical_planner/param/follow_when_forced',
            self._cb_follow_when_forced, qos_reliable,
            callback_group=self._sub_cb_group)

        # ── Publishers ───────────────────────────────────────────────
        self.pub_traj = self.create_publisher(
            ReferencePath,
            '/flyeagle/a2rl/tactical_planner/trajectory',
            qos_reliable)

        self.pub_status = self.create_publisher(
            String, '/tactical_planner/status', qos_reliable)

        self.pub_fsm = self.create_publisher(
            String, '/tactical_planner/fsm_state', qos_reliable)

        self.pub_domain = self.create_publisher(
            Float32MultiArray, '/tactical_planner/corridor', qos_reliable)

        self.pub_thresholds = self.create_publisher(
            String, '/tactical_planner/thresholds', qos_reliable)

        # ── Timer-driven planning ────────────────────────────────────
        timer_period = 1.0 / max(timer_hz, 1.0)
        self.timer = self.create_timer(
            timer_period, self._timer_callback,
            callback_group=self._timer_cb_group)

        self.get_logger().info(
            f'\033[1;33m[TacticalPlannerNode] started  '
            f'policy={policy_type}  force_side={force_side}  '
            f'timer={timer_hz:.0f}Hz  track={track_name}\033[0m')

    # ==================================================================
    # Subscriber Callbacks
    # ==================================================================

    def _cb_localization(self, msg: Localization):
        with self._data_lock:
            self._loc_msg = msg
            self._loc_stamp = self.get_clock().now()

    def _cb_ego_state(self, msg: EgoState):
        with self._data_lock:
            self._ego_msg = msg
            self._ego_stamp = self.get_clock().now()

    def _cb_opponents(self, msg: GroundTruthArray):
        with self._data_lock:
            self._opp_detections = list(msg.vehicles)
            self._opp_stamp = self.get_clock().now()

    def _cb_force_side(self, msg: String):
        val = msg.data.strip().lower()
        side = None if val in ('none', '') else val
        if hasattr(self.policy, 'set_force_side'):
            self.policy.set_force_side(side)
        self.get_logger().info(f'[param] force_side ← {side}')

    def _cb_follow_when_forced(self, msg: Bool):
        if hasattr(self.policy, 'set_follow_when_forced'):
            self.policy.set_follow_when_forced(msg.data)
        self.get_logger().info(f'[param] follow_when_forced ← {msg.data}')

    # ==================================================================
    # Ego state construction from ROS2 messages
    # ==================================================================

    def _build_ego_state(self) -> Optional[dict]:
        """Convert latest Localization + EgoState messages to ego_state dict.

        CRITICAL: The ACADOS OCP model uses Frenet-frame accelerations:
          ax = longitudinal accel along track tangent  (ds/dt derivative)
          ay = lateral accel perpendicular to tangent   (centripetal component)
        The EgoState message provides body-frame accelerations, so we must
        rotate them from body frame → global frame → Frenet frame.
        """
        with self._data_lock:
            lc = self._loc_msg
            eg = self._ego_msg
        if lc is None or eg is None:
            return None

        x = float(lc.position.x)
        y = float(lc.position.y)
        z = float(lc.position.z)
        yaw = float(lc.orientation_ypr.z)

        vx = float(eg.velocity.x)
        vy = float(eg.velocity.y)
        V = math.sqrt(vx * vx + vy * vy)

        # --- Body-frame accelerations from IMU / observer ---
        ax_body = float(eg.acceleration.x)  # forward in body
        ay_body = float(eg.acceleration.y)  # leftward in body

        # Convert to Frenet via track_handler
        try:
            sn = self.track_handler.cartesian2sn(x, y)
            s = float(sn[0])
            n = float(sn[1])
        except Exception:
            if self._ego_state is not None:
                return self._ego_state   # keep last valid
            return None

        # chi = heading deviation from road tangent
        Omega_z = float(np.interp(
            s, self.track_handler.s, self.track_handler.Omega_z,
            period=self.track_handler.s[-1]))
        road_heading = float(np.interp(
            s, self.track_handler.s, self.track_handler.psi,
            period=self.track_handler.s[-1]))

        chi = _wrap_angle(yaw - road_heading)

        s_dot = V * math.cos(chi) / max(1.0 - n * Omega_z, 0.01)
        n_dot = V * math.sin(chi)

        # --- Convert body-frame accel → Frenet accel ---
        # Body frame is rotated by `chi` relative to Frenet tangent.
        #   ax_frenet = ax_body * cos(chi) - ay_body * sin(chi)
        #   ay_frenet = ax_body * sin(chi) + ay_body * cos(chi)
        # This is the same rotation used in the OCP model.
        cos_chi = math.cos(chi)
        sin_chi = math.sin(chi)
        ax_frenet = ax_body * cos_chi - ay_body * sin_chi
        ay_frenet = ax_body * sin_chi + ay_body * cos_chi

        return {
            's': s, 'n': n, 'V': V,
            'chi': chi, 'ax': ax_frenet, 'ay': ay_frenet,
            'x': x, 'y': y, 'z': z,
            's_dot': s_dot, 'n_dot': n_dot,
            'time_ns': int(lc.timestamp.nanoseconds),
        }

    # ==================================================================
    # Opponent state construction
    # ==================================================================

    def _build_opp_states(self) -> list:
        """Convert GroundTruth detections to opponent state dicts.

        GroundTruth message fields (from autonoma_msgs):
          del_x, del_y, del_z : position offset in **ego** body frame [m]
          vx, vy, vz          : velocity in **opponent's own** body frame [m/s]
          yaw                  : **global absolute** heading [rad from North, CCW+]
          car_num              : opponent vehicle ID

        We convert everything to global (x, y, V_abs, yaw_abs) then to
        Frenet (s, n, V, chi).
        """
        with self._data_lock:
            detections = list(self._opp_detections)
        if self._ego_state is None or not detections:
            return []

        ego_x = self._ego_state['x']
        ego_y = self._ego_state['y']
        ego_V = self._ego_state['V']
        # Use localization yaw directly (most reliable source)
        if self._loc_msg is not None:
            ego_yaw = float(self._loc_msg.orientation_ypr.z)
        else:
            ego_yaw = math.atan2(
                self._ego_state.get('n_dot', 0.0),
                self._ego_state.get('s_dot', 1.0),
            )

        cos_y = math.cos(ego_yaw)
        sin_y = math.sin(ego_yaw)

        opp_list = []
        for i, gt in enumerate(detections):
            # --- Position: del_x/del_y are in ego body frame ---
            dx = float(gt.del_x)
            dy = float(-gt.del_y)  # negate: rightward→leftward (same as C++)

            # ego-local → global position
            gx = ego_x + cos_y * dx - sin_y * dy
            gy = ego_y + sin_y * dx + cos_y * dy

            # --- Opponent heading (global absolute) ---
            # gt.yaw is global absolute heading (rad from North, CCW+)
            opp_yaw_global = float(gt.yaw)

            # --- Opponent speed ---
            # gt.vx/vy are in opponent's OWN body frame.
            # Rotate to global frame using opponent's global yaw.
            opp_vx_body = float(gt.vx)   # forward in opp body
            opp_vy_body = float(gt.vy)   # leftward in opp body
            cos_opp = math.cos(opp_yaw_global)
            sin_opp = math.sin(opp_yaw_global)
            opp_vx_global = cos_opp * opp_vx_body - sin_opp * opp_vy_body
            opp_vy_global = sin_opp * opp_vx_body + cos_opp * opp_vy_body
            opp_V = math.sqrt(opp_vx_global ** 2 + opp_vy_global ** 2)

            try:
                sn = self.track_handler.cartesian2sn(gx, gy)
                opp_s = float(sn[0])
                opp_n = float(sn[1])
            except Exception:
                continue

            # Opponent chi = deviation from road tangent at opp position
            road_heading_opp = float(np.interp(
                opp_s, self.track_handler.s, self.track_handler.psi,
                period=self.track_handler.s[-1]))
            opp_chi = _wrap_angle(opp_yaw_global - road_heading_opp)

            opp_list.append({
                'id': int(gt.car_num) if hasattr(gt, 'car_num') else i,
                's': opp_s, 'n': opp_n, 'V': opp_V,
                'chi': opp_chi, 'ax': 0.0, 'ay': 0.0,
                'x': gx, 'y': gy, 'z': 0.0,
                'tactic': 'unknown',
                'target_n_offset': 0.0,
            })

        return opp_list

    # ==================================================================
    # Timer callback — main planning loop
    # ==================================================================

    def _timer_callback(self):
        t0 = time.time()

        # 1) build ego state
        ego = self._build_ego_state()
        if ego is None:
            if self._step % 40 == 0:  # throttle warn
                loc_age = (self.get_clock().now() - self._loc_stamp).nanoseconds * 1e-9
                ego_age = (self.get_clock().now() - self._ego_stamp).nanoseconds * 1e-9
                self.get_logger().warning(
                    f'[Tactical] Waiting for ego data  loc_age={loc_age:.1f}s  '
                    f'ego_age={ego_age:.1f}s')
            self._step += 1
            return
        self._ego_state = ego

        # 2) build opponent states
        opp_states = self._build_opp_states()

        # Provide constant-velocity predictions over the planning horizon.
        # The test script (sim_tactical.py) uses OpponentVehicle.predict()
        # which generates a multi-step trajectory.  For real-time we
        # approximate with straight-line constant-speed prediction in
        # Frenet coordinates, matching the carver's expectation.
        n_pred_steps = 10  # ~1.25s at dt=0.125
        pred_dt = self.cfg.dt
        for os_d in opp_states:
            opp_V = os_d['V']
            opp_s = os_d['s']
            opp_n = os_d['n']
            opp_chi = os_d.get('chi', 0.0)

            # Compute opp s_dot in Frenet
            opp_Omega_z = float(np.interp(
                opp_s, self.track_handler.s, self.track_handler.Omega_z,
                period=self.track_handler.s[-1]))
            opp_s_dot = opp_V * math.cos(opp_chi) / max(1.0 - opp_n * opp_Omega_z, 0.01)
            opp_n_dot = opp_V * math.sin(opp_chi)

            pred_s = np.array([opp_s + opp_s_dot * pred_dt * k
                               for k in range(n_pred_steps)])
            pred_s = pred_s % self.track_handler.s[-1]
            pred_n = np.array([opp_n + opp_n_dot * pred_dt * k
                               for k in range(n_pred_steps)])
            # Convert to Cartesian for carver
            try:
                pred_xy = self.track_handler.sn2cartesian(pred_s, pred_n)
                if pred_xy.ndim == 1:
                    pred_xy = pred_xy.reshape(1, -1)
                os_d['pred_s'] = pred_s
                os_d['pred_n'] = pred_n
                os_d['pred_x'] = pred_xy[:, 0]
                os_d['pred_y'] = pred_xy[:, 1]
            except Exception:
                os_d['pred_s'] = np.array([opp_s])
                os_d['pred_n'] = np.array([opp_n])
                os_d['pred_x'] = np.array([os_d['x']])
                os_d['pred_y'] = np.array([os_d['y']])

        # 3) build observation
        obs = build_observation(
            ego_state=ego,
            opponents=opp_states,
            track_handler=self.track_handler,
            p2p_state=self.p2p.get_state_vector(),
            prev_action_array=self.prev_action.to_array(),
            planner_healthy=self.planner.planner_healthy,
            cfg=self.cfg,
        )

        # 4) tactical action
        action = self.policy.act(obs)

        # 4.5) feed carver overtake_ready back
        if hasattr(self.policy, 'set_overtake_ready'):
            self.policy.set_overtake_ready(self.a2rl_carver.overtake_ready)

        # 5) guidance
        guidance = self.tactical_mapper.map(
            action, obs, N_stages=self.cfg.N_steps_acados)

        c_mode = _CARVER_MODE_MAP.get(
            getattr(self.policy, 'carver_mode_str', 'follow'),
            CarverMode.FOLLOW)
        c_side = getattr(self.policy, 'carver_side', None)

        horizon_m = self.cfg.optimization_horizon_m
        ds = horizon_m / self.cfg.N_steps_acados
        follow_opps = getattr(self.policy, 'follow_when_forced', True)

        carver_guidance = self.a2rl_carver.construct_guidance(
            ego,
            opp_states,
            self.cfg.N_steps_acados,
            ds,
            mode=c_mode,
            shadow_side=c_side,
            overtake_side=c_side,
            prev_trajectory=self.planner._prev_trajectory,
            planner_healthy=self.planner.planner_healthy,
            follow_opponents=follow_opps,
        )

        if carver_guidance.n_left_override is not None:
            guidance.n_left_override = carver_guidance.n_left_override
        if carver_guidance.n_right_override is not None:
            guidance.n_right_override = carver_guidance.n_right_override
        if carver_guidance.speed_cap < guidance.speed_cap:
            guidance.speed_cap = carver_guidance.speed_cap
        if carver_guidance.speed_scale < guidance.speed_scale:
            guidance.speed_scale = carver_guidance.speed_scale

        # 6) plan
        trajectory = self.planner.plan(ego, guidance)
        t_plan = time.time() - t0

        # 7) publish trajectory
        self._publish_trajectory(trajectory, ego)

        # 8) publish status / FSM / domain / thresholds
        self._publish_status(trajectory, action, c_mode, c_side, t_plan,
                             guidance)
        self._publish_fsm_state(action, guidance)
        self._publish_thresholds()
        self._publish_feasible_domain(guidance, ego)
        self._publish_corridor_data(guidance, ego)

        self.prev_action = action
        self._step += 1

        if self._step % 40 == 0:
            ph = getattr(self.policy, 'debug_info', {}).get('phase', '?')
            gap = getattr(self.policy, 'debug_info', {}).get('gap', None)
            gap_s = f'{gap:.1f}' if gap else 'N/A'
            dbg = getattr(self.planner, 'debug_log', {})
            fallback = dbg.get('used_fallback', False)
            consec_fail = dbg.get('_consecutive_failures', 0)
            v_max_eff = guidance.get_effective_v_max(
                self.params['vehicle_params'].get('v_max', 90.0), self.cfg)
            traj_v0 = trajectory['V'][0] if len(trajectory['V']) > 0 else 0.0
            traj_vN = trajectory['V'][-1] if len(trajectory['V']) > 0 else 0.0
            self.get_logger().info(
                f'[{self._step:5d}] s={ego["s"]:.1f} n={ego["n"]:.2f} '
                f'V={ego["V"]:.1f} chi={ego["chi"]:.3f} '
                f'ax={ego["ax"]:.2f} ay={ego["ay"]:.2f} | '
                f'{ph} gap={gap_s} | '
                f'{c_mode.name} {c_side or ""} | '
                f'traj_V=[{traj_v0:.1f}..{traj_vN:.1f}] vmax={v_max_eff:.1f} '
                f'healthy={self.planner.planner_healthy} '
                f'fallback={fallback} fail={consec_fail} | '
                f'{t_plan*1000:.0f}ms')

    # ==================================================================
    # Publishers
    # ==================================================================

    def _publish_trajectory(self, trajectory: dict, ego: dict):
        """Pack trajectory into a2rl_bs_msgs/ReferencePath and publish.

        Encoding convention (spare fields carry Frenet state so that the
        subscriber can avoid lossy cartesian→Frenet back-conversion):

          orientation_ypr.x  = x_global
          orientation_ypr.y  = y_global
          orientation_ypr.z  = s  (Frenet arc-length)
          velocity_linear.x  = speed
          velocity_linear.y  = yaw (global heading)
          velocity_angular.x = n  (Frenet lateral offset)
          velocity_angular.y = chi (Frenet heading deviation)
          velocity_angular.z = yaw_rate
          acceleration.x     = ax
          acceleration.y     = ay
          acceleration.z     = t  (time stamp of this waypoint)
        """
        msg = ReferencePath()
        msg.timestamp.nanoseconds = ego.get('time_ns', 0)

        # Use the *actual* time spacing from the planner output
        t_arr = trajectory['t']
        if len(t_arr) >= 2:
            msg.path_time_discretization_s = float(t_arr[1] - t_arr[0])
        else:
            msg.path_time_discretization_s = float(
                self.cfg.planning_horizon / max(len(t_arr), 1))

        msg.origin_position.x = float(ego['x'])
        msg.origin_position.y = float(ego['y'])
        msg.origin_position.z = float(ego['z'])
        msg.origin_orientation_ypr.z = float(
            self._loc_msg.orientation_ypr.z if self._loc_msg else 0.0)

        N = min(len(trajectory['x']), _MAX_PATH_POINTS)
        for i in range(N):
            pt = CartesianFrameState()
            pt.timestamp.nanoseconds = ego.get('time_ns', 0)

            # position: local frame (will be recomputed in planner.cpp)
            pt.position.x = float(trajectory['x'][i])
            pt.position.y = float(trajectory['y'][i])
            pt.position.z = float(trajectory.get('z', np.zeros(N))[i])

            # Convention matching case 14/15 in planner.cpp:
            #   orientation_ypr.x = x_global
            #   orientation_ypr.y = y_global
            #   orientation_ypr.z = s (Frenet)
            pt.orientation_ypr.x = float(trajectory['x'][i])
            pt.orientation_ypr.y = float(trajectory['y'][i])
            pt.orientation_ypr.z = float(trajectory['s'][i])

            # velocity_linear.x = speed
            # velocity_linear.y = yaw (heading)
            V_i = float(trajectory['V'][i])
            pt.velocity_linear.x = V_i

            # Compute heading from (s, chi) → global yaw
            s_i = float(trajectory['s'][i])
            chi_i = float(trajectory['chi'][i])
            road_yaw = float(np.interp(
                s_i, self.track_handler.s, self.track_handler.psi,
                period=self.track_handler.s[-1]))
            yaw_i = road_yaw + chi_i
            pt.velocity_linear.y = yaw_i

            # Frenet state in spare angular-velocity slots
            n_i = float(trajectory['n'][i])
            pt.velocity_angular.x = n_i      # n
            pt.velocity_angular.y = chi_i     # chi

            # velocity_angular.z = yaw_rate ≈ curvature × speed
            Omega_z = float(np.interp(
                s_i, self.track_handler.s, self.track_handler.Omega_z,
                period=self.track_handler.s[-1]))
            kappa = Omega_z / max(1.0 - n_i * Omega_z, 0.01)
            yaw_rate = kappa * V_i
            pt.velocity_angular.z = yaw_rate

            # acceleration.x = ax,  acceleration.y = ay  (Frenet accel)
            ax_arr = trajectory.get('ax', np.zeros(N))
            ay_arr = trajectory.get('ay', np.zeros(N))
            pt.acceleration.x = float(ax_arr[i])
            pt.acceleration.y = float(ay_arr[i])
            pt.acceleration.z = float(t_arr[i])   # time

            msg.path.append(pt)

        self.pub_traj.publish(msg)

    def _publish_status(self, trajectory, action, c_mode, c_side, t_plan,
                        guidance=None):
        """Publish JSON status string.

        Includes corridor bounds (n_left_override / n_right_override) so
        that sim_env_node can forward them to the TacticalVisualizer.
        """
        status = {
            'step': self._step,
            'planner_ok': self.planner.planner_healthy,
            'tactic': action.discrete_tactic.name,
            'carver_mode': c_mode.name,
            'carver_side': c_side,
            'overtake_ready': self.a2rl_carver.overtake_ready,
            'plan_ms': round(t_plan * 1000, 1),
            'n_points': len(trajectory['x']),
        }
        # Corridor bounds for visualizer
        if guidance is not None:
            if getattr(guidance, 'n_left_override', None) is not None:
                status['n_left_override'] = [
                    round(float(v), 4) for v in guidance.n_left_override]
            if getattr(guidance, 'n_right_override', None) is not None:
                status['n_right_override'] = [
                    round(float(v), 4) for v in guidance.n_right_override]
        msg = String()
        msg.data = json.dumps(status)
        self.pub_status.publish(msg)

    def _publish_fsm_state(self, action, guidance):
        """Publish detailed FSM/policy state for diagnostics."""
        info = getattr(self.policy, 'debug_info', {})
        fsm = {
            'phase': info.get('phase', 'N/A'),
            'target_id': info.get('target_id'),
            'gap': info.get('gap'),
            'locked_side': info.get('locked_side'),
            'phase_time': info.get('phase_time'),
            'tactic': action.discrete_tactic.name,
            'aggressiveness': round(action.aggressiveness, 3),
            'terminal_n': round(guidance.terminal_n_target, 3),
            'speed_cap': round(guidance.speed_cap, 1),
            'speed_scale': round(guidance.speed_scale, 3),
            'safety_distance': round(guidance.safety_distance, 3),
        }
        msg = String()
        msg.data = json.dumps(fsm)
        self.pub_fsm.publish(msg)

    def _publish_thresholds(self):
        """Publish tactical thresholds (constants from config)."""
        cfg = self.cfg
        thresholds = {
            'chase_gap': getattr(cfg, 'chase_gap', 15.0),
            'overtake_gap': getattr(cfg, 'overtake_gap', 12.0),
            'abort_gap': getattr(cfg, 'abort_gap', 25.0),
            'follow_dist': getattr(cfg, 'follow_distance', 8.0),
            'safety_dist': cfg.safety_distance_default,
            'vehicle_length': cfg.vehicle_length,
            'vehicle_width': cfg.vehicle_width,
            'lateral_safety': getattr(cfg, 'lateral_safety', 2.0),
        }
        msg = String()
        msg.data = json.dumps(thresholds)
        self.pub_thresholds.publish(msg)

    def _publish_feasible_domain(self, guidance, ego):
        """DEPRECATED: feasible domain visualization moved to planner_cvxopt.
        Kept as no-op stub for compatibility."""
        pass

    def _publish_corridor_data(self, guidance, ego):
        """Publish corridor boundary data as Float32MultiArray to planner_cvxopt.

        The planner_cvxopt node receives this and renders the feasible-domain
        MarkerArray using its own track data (yasnorth_map_sim.csv), ensuring
        the visualization is consistent with /flyeagle/global_path.

        Format  (all float32):
          data[0]         = N   (number of sample points along horizon)
          data[1..3N]     = active_left  polyline: x0,y0,z0, x1,y1,z1, ...
          data[3N+1..6N]  = active_right polyline: x0,y0,z0, x1,y1,z1, ...
          data[6N+1..9N]  = base_left    polyline: x0,y0,z0, x1,y1,z1, ...
          data[9N+1..12N] = base_right   polyline: x0,y0,z0, x1,y1,z1, ...
        """
        track_len = float(self.track_handler.s[-1])
        s0 = float(ego['s'])
        ds = float(self.cfg.optimization_horizon_m / self.cfg.N_steps_acados)

        n_samples = int(self.cfg.N_steps_acados) + 1
        s_arr_raw = s0 + np.arange(n_samples) * ds
        s_arr = s_arr_raw % track_len

        base_left = np.interp(
            s_arr, self.track_handler.s, self.track_handler.w_tr_left,
            period=track_len)
        base_right = np.interp(
            s_arr, self.track_handler.s, self.track_handler.w_tr_right,
            period=track_len)

        w_left = getattr(guidance, 'n_left_override', None)
        w_right = getattr(guidance, 'n_right_override', None)

        dom_left = np.asarray(base_left, dtype=float)
        dom_right = np.asarray(base_right, dtype=float)

        if w_left is not None:
            src = np.asarray(w_left, dtype=float)
            dom_left = np.interp(
                np.linspace(0.0, 1.0, n_samples),
                np.linspace(0.0, 1.0, len(src)), src)
        if w_right is not None:
            src = np.asarray(w_right, dtype=float)
            dom_right = np.interp(
                np.linspace(0.0, 1.0, n_samples),
                np.linspace(0.0, 1.0, len(src)), src)

        # Convert all 4 boundary lines to XYZ
        def _sn_to_xyz_array(s_arr, n_arr, z_offset=0.30):
            out = []
            for s_v, n_v in zip(s_arr, n_arr):
                try:
                    xyz = self.track_handler.sn2cartesian(float(s_v), float(n_v))
                    out.extend([float(xyz[0]), float(xyz[1]),
                                float(xyz[2]) + z_offset])
                except Exception:
                    out.extend([0.0, 0.0, 0.0])
            return out

        buf = [float(n_samples)]
        buf.extend(_sn_to_xyz_array(s_arr, dom_left))
        buf.extend(_sn_to_xyz_array(s_arr, dom_right))
        buf.extend(_sn_to_xyz_array(s_arr, base_left))
        buf.extend(_sn_to_xyz_array(s_arr, base_right))

        msg = Float32MultiArray()
        msg.data = [float(v) for v in buf]
        self.pub_domain.publish(msg)


# ======================================================================
# Helpers
# ======================================================================

def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def main(args=None):
    rclpy.init(args=args)
    node = TacticalPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

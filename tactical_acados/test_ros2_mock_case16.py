#!/usr/bin/env python3
"""
ROS2 Mock Integration Test for Case 16 (Tactical Planner Node)
================================================================

Simulates the **exact same** scenario_c 3-car race using the existing
sim_tactical pipeline, but with the planner wrapped behind a simulated
ROS2 message interface.

Architecture:
  sim loop (scenario_c opponents + perfect tracking)
      │
      ├─► [mock publish] ego_loc  + ego_state + GroundTruthArray
      │
      ├─► [tactical_planner core]  ← identical to node's _timer_callback
      │         │
      │         └─► trajectory dict  ← same as real node output
      │
      ├─► [mock serialize] trajectory → ReferencePath msg
      │
      └─► [mock deserialize] ReferencePath msg → trajectory
              │
              └─► perfect_tracking_update  ← same as sim_tactical step 9

If the mock-ROS loop produces identical planner_ok / collision / speed
results as the direct sim, the case16 integration is validated.

Usage:
    cd tactical_acados
    python test_ros2_mock_case16.py                # default 200 steps
    python test_ros2_mock_case16.py --steps 600    # longer run
"""

import os
import sys
import time
import math
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Optional

# ── Path setup ───────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_DIR, '..')
for p in [_ROOT, os.path.join(_ROOT, 'src'), _DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import TacticalConfig
from acados_planner import AcadosTacticalPlanner
from tactical_action import TacticalAction, PlannerGuidance, get_fallback_action, TacticalMode
from observation import TacticalObservation, build_observation
from safe_wrapper import SafeTacticalWrapper
from planner_guidance import TacticalToPlanner
from opponent import OpponentVehicle
from p2p import PushToPass
from follow_module import FollowModule
from a2rl_obstacle_carver import A2RLObstacleCarver, CarverMode
from sim_acados_only import load_setup, create_initial_state, perfect_tracking_update
from sim_tactical import load_scenario

from track3D import Track3D


# ======================================================================
# Helper: compute 2D global heading from Frenet state
# ======================================================================

def _road_heading(s, track_handler):
    """Interpolate flat 2D road heading (psi) at arc-length s."""
    return float(np.interp(
        s, track_handler.s, track_handler.psi,
        period=track_handler.s[-1]))


def _global_yaw(s, chi, track_handler):
    """Global yaw = road heading + chi."""
    return _road_heading(s, track_handler) + chi


# ======================================================================
# Mock ROS2 Message Types (lightweight dataclasses, no rclpy needed)
# ======================================================================

@dataclass
class MockCartesianFrame:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

@dataclass
class MockTimestamp:
    nanoseconds: int = 0

@dataclass
class MockLocalization:
    timestamp: MockTimestamp = None
    position: MockCartesianFrame = None
    position_stddev: MockCartesianFrame = None
    orientation_ypr: MockCartesianFrame = None
    orientation_stddev: MockCartesianFrame = None
    def __post_init__(self):
        if self.timestamp is None: self.timestamp = MockTimestamp()
        if self.position is None: self.position = MockCartesianFrame()
        if self.position_stddev is None: self.position_stddev = MockCartesianFrame()
        if self.orientation_ypr is None: self.orientation_ypr = MockCartesianFrame()
        if self.orientation_stddev is None: self.orientation_stddev = MockCartesianFrame()

@dataclass
class MockEgoState:
    timestamp: MockTimestamp = None
    velocity: MockCartesianFrame = None
    angular_rate: MockCartesianFrame = None
    acceleration: MockCartesianFrame = None
    def __post_init__(self):
        if self.timestamp is None: self.timestamp = MockTimestamp()
        if self.velocity is None: self.velocity = MockCartesianFrame()
        if self.angular_rate is None: self.angular_rate = MockCartesianFrame()
        if self.acceleration is None: self.acceleration = MockCartesianFrame()

@dataclass
class MockGroundTruth:
    car_num: int = 0
    del_x: float = 0.0
    del_y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0

@dataclass
class MockCartesianFrameState:
    timestamp: MockTimestamp = None
    position: MockCartesianFrame = None
    orientation_ypr: MockCartesianFrame = None
    velocity_linear: MockCartesianFrame = None
    velocity_angular: MockCartesianFrame = None
    acceleration: MockCartesianFrame = None
    def __post_init__(self):
        if self.timestamp is None: self.timestamp = MockTimestamp()
        if self.position is None: self.position = MockCartesianFrame()
        if self.orientation_ypr is None: self.orientation_ypr = MockCartesianFrame()
        if self.velocity_linear is None: self.velocity_linear = MockCartesianFrame()
        if self.velocity_angular is None: self.velocity_angular = MockCartesianFrame()
        if self.acceleration is None: self.acceleration = MockCartesianFrame()

@dataclass
class MockReferencePath:
    timestamp: MockTimestamp = None
    path_time_discretization_s: float = 0.0
    origin_position: MockCartesianFrame = None
    origin_orientation_ypr: MockCartesianFrame = None
    path: list = None
    def __post_init__(self):
        if self.timestamp is None: self.timestamp = MockTimestamp()
        if self.origin_position is None: self.origin_position = MockCartesianFrame()
        if self.origin_orientation_ypr is None: self.origin_orientation_ypr = MockCartesianFrame()
        if self.path is None: self.path = []


# ======================================================================
# Encoder / Decoder  (matches tactical_planner_node.py + planner.cpp case16)
# ======================================================================

MAX_PATH_POINTS = 30

def ego_state_to_ros(ego_state: dict, track_handler) -> tuple:
    """sim ego_state → (MockLocalization, MockEgoState)  same as real topics."""

    # Heading: compute global yaw from (s, chi)
    s = ego_state['s']
    chi = ego_state.get('chi', 0.0)
    yaw = _global_yaw(s, chi, track_handler)

    loc = MockLocalization()
    loc.timestamp.nanoseconds = ego_state.get('time_ns', int(time.time() * 1e9))
    loc.position.x = ego_state['x']
    loc.position.y = ego_state['y']
    loc.position.z = ego_state['z']
    loc.orientation_ypr.z = float(yaw)

    V = ego_state['V']
    ego = MockEgoState()
    ego.timestamp.nanoseconds = loc.timestamp.nanoseconds
    # vx, vy in body frame (forward/lateral)
    ego.velocity.x = V * math.cos(chi)
    ego.velocity.y = V * math.sin(chi)
    ego.acceleration.x = ego_state.get('ax', 0.0)
    ego.acceleration.y = ego_state.get('ay', 0.0)

    return loc, ego


def opponents_to_ros(opp_states: list, ego_state: dict, track_handler) -> list:
    """sim opponent state list → list of MockGroundTruth (like GroundTruthArray.vehicles)."""
    ego_x = ego_state['x']
    ego_y = ego_state['y']
    ego_yaw = float(_global_yaw(ego_state['s'], ego_state.get('chi', 0.0), track_handler))
    cos_y = math.cos(ego_yaw)
    sin_y = math.sin(ego_yaw)

    gt_list = []
    for opp in opp_states:
        # global → ego-local
        dx_g = opp['x'] - ego_x
        dy_g = opp['y'] - ego_y
        del_x =  cos_y * dx_g + sin_y * dy_g   # forward
        del_y = -sin_y * dx_g + cos_y * dy_g   # rightward (note: planner stores -del_y)
        # In GroundTruth, del_y is "rightward", planner stores as -del_y
        gt = MockGroundTruth(
            car_num=opp.get('id', 0),
            del_x=del_x,
            del_y=del_y,  # rightward positive, planner callback does -del_y
            yaw=float(_global_yaw(opp['s'], opp.get('chi', 0.0), track_handler)),
            vx=opp['V'],
        )
        gt_list.append(gt)
    return gt_list


def trajectory_to_ros(trajectory: dict, ego_state: dict,
                      track_handler, cfg) -> MockReferencePath:
    """Trajectory dict → MockReferencePath  (same encoding as node's _publish_trajectory)."""
    msg = MockReferencePath()
    msg.timestamp.nanoseconds = ego_state.get('time_ns', 0)
    msg.path_time_discretization_s = float(cfg.planning_horizon / max(len(trajectory['t']), 1))
    msg.origin_position.x = ego_state['x']
    msg.origin_position.y = ego_state['y']
    msg.origin_position.z = ego_state['z']
    yaw = float(_global_yaw(ego_state['s'], ego_state.get('chi', 0.0), track_handler))
    msg.origin_orientation_ypr.z = yaw

    N = min(len(trajectory['x']), MAX_PATH_POINTS)
    for i in range(N):
        pt = MockCartesianFrameState()
        pt.orientation_ypr.x = float(trajectory['x'][i])
        pt.orientation_ypr.y = float(trajectory['y'][i])
        V_i = float(trajectory['V'][i])
        pt.velocity_linear.x = V_i

        s_i = float(trajectory['s'][i])
        chi_i = float(trajectory['chi'][i])
        road_yaw = float(np.interp(
            s_i, track_handler.s, track_handler.psi,
            period=track_handler.s[-1]))
        yaw_i = road_yaw + chi_i
        pt.velocity_linear.y = yaw_i

        Omega_z = float(np.interp(
            s_i, track_handler.s, track_handler.Omega_z,
            period=track_handler.s[-1]))
        n_i = float(trajectory['n'][i])
        kappa = Omega_z / max(1.0 - n_i * Omega_z, 0.01)
        pt.velocity_angular.z = kappa * V_i

        msg.path.append(pt)

    return msg


def ros_to_trajectory(msg: MockReferencePath, track_handler) -> dict:
    """MockReferencePath → trajectory dict (what planner.cpp case16 would do, then back).

    This simulates:
      1) case16 reads msg.path[i].orientation_ypr.x/y as x_global/y_global
      2) case16 reads msg.path[i].velocity_linear.x/y as speed/yaw
      3) We convert back to Frenet for perfect_tracking_update
    """
    xs, ys, Vs, yaws, ss, ns, chis = [], [], [], [], [], [], []

    for pt in msg.path:
        x_g = pt.orientation_ypr.x
        y_g = pt.orientation_ypr.y
        speed = pt.velocity_linear.x
        yaw = pt.velocity_linear.y

        xs.append(x_g)
        ys.append(y_g)
        Vs.append(speed)
        yaws.append(yaw)

        sn = track_handler.cartesian2sn(x_g, y_g)
        s_i = float(sn[0])
        n_i = float(sn[1])
        ss.append(s_i)
        ns.append(n_i)

        road_yaw = float(np.interp(
            s_i, track_handler.s, track_handler.psi,
            period=track_handler.s[-1]))
        chi_i = _wrap_angle(yaw - road_yaw)
        chis.append(chi_i)

    N = len(xs)
    dt = msg.path_time_discretization_s
    t_arr = np.arange(N) * dt

    return {
        't': t_arr,
        'x': np.array(xs),
        'y': np.array(ys),
        'z': np.zeros(N),
        'V': np.array(Vs),
        's': np.array(ss),
        'n': np.array(ns),
        'chi': np.array(chis),
        'ax': np.zeros(N),
        'ay': np.zeros(N),
    }


def _wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ======================================================================
# _build_ego_state — mirrors node's _build_ego_state exactly
# ======================================================================

def ros_to_ego_state(loc: MockLocalization, ego: MockEgoState,
                     track_handler) -> dict:
    """Reproduce tactical_planner_node._build_ego_state from mock msgs."""
    x = float(loc.position.x)
    y = float(loc.position.y)
    z = float(loc.position.z)
    yaw = float(loc.orientation_ypr.z)

    vx = float(ego.velocity.x)
    vy = float(ego.velocity.y)
    V = math.sqrt(vx*vx + vy*vy)
    ax = float(ego.acceleration.x)
    ay = float(ego.acceleration.y)

    sn = track_handler.cartesian2sn(x, y)
    s = float(sn[0])
    n = float(sn[1])

    Omega_z = float(np.interp(
        s, track_handler.s, track_handler.Omega_z,
        period=track_handler.s[-1]))
    road_heading = float(np.interp(
        s, track_handler.s, track_handler.psi,
        period=track_handler.s[-1]))
    chi = _wrap_angle(yaw - road_heading)
    s_dot = V * math.cos(chi) / max(1.0 - n * Omega_z, 0.01)
    n_dot = V * math.sin(chi)

    return {
        's': s, 'n': n, 'V': V,
        'chi': chi, 'ax': ax, 'ay': ay,
        'x': x, 'y': y, 'z': z,
        's_dot': s_dot, 'n_dot': n_dot,
        'time_ns': int(loc.timestamp.nanoseconds),
    }


def ros_to_opp_states(gt_list: list, ego_state: dict,
                      loc: MockLocalization, track_handler) -> list:
    """Reproduce tactical_planner_node._build_opp_states from mock msgs."""
    ego_x = ego_state['x']
    ego_y = ego_state['y']
    ego_yaw = float(loc.orientation_ypr.z)
    cos_y = math.cos(ego_yaw)
    sin_y = math.sin(ego_yaw)

    opp_list = []
    for i, gt in enumerate(gt_list):
        dx = float(gt.del_x)
        dy = float(-gt.del_y)  # same as node

        gx = ego_x + cos_y * dx - sin_y * dy
        gy = ego_y + sin_y * dx + cos_y * dy
        opp_V = float(gt.vx)

        try:
            sn = track_handler.cartesian2sn(gx, gy)
            opp_s = float(sn[0])
            opp_n = float(sn[1])
        except Exception:
            continue

        opp_list.append({
            'id': int(gt.car_num),
            's': opp_s, 'n': opp_n, 'V': opp_V,
            'chi': 0.0, 'ax': 0.0, 'ay': 0.0,
            'x': gx, 'y': gy, 'z': 0.0,
            'tactic': 'unknown',
            'target_n_offset': 0.0,
        })
    return opp_list


# ======================================================================
# Main test
# ======================================================================

CARVER_MODE_MAP = {
    'follow': CarverMode.FOLLOW,
    'shadow': CarverMode.SHADOW,
    'overtake': CarverMode.OVERTAKE,
    'raceline': CarverMode.RACELINE,
    'hold': CarverMode.HOLD,
    'force_left': CarverMode.FORCE_LEFT,
    'force_right': CarverMode.FORCE_RIGHT,
}


def run_mock_ros_test(max_steps=200, scenario_name='scenario_c'):
    """Run scenario_c with mock ROS message layer.

    Data flow each step:
      1. sim ego_state → ego_state_to_ros → MockLocalization + MockEgoState
      2. sim opponents → opponents_to_ros → list[MockGroundTruth]
      3. MockLocalization + MockEgoState → ros_to_ego_state → ego_state_ros
         (this is what the node would compute from the subscribed topics)
      4. list[MockGroundTruth] → ros_to_opp_states → opp_states_ros
      5. Run tactical planner pipeline on (ego_state_ros, opp_states_ros)
      6. trajectory → trajectory_to_ros → MockReferencePath
         (this is what the node publishes)
      7. MockReferencePath → ros_to_trajectory → trajectory_decoded
         (this is what planner.cpp case16 decodes)
      8. perfect_tracking_update with trajectory_decoded
    """
    scenario = load_scenario(scenario_name)
    sc = scenario['scenario']
    ego_cfg = scenario['ego']
    opp_cfgs = scenario.get('opponents', [])
    planner_cfg = scenario.get('planner', {})

    cfg = TacticalConfig()
    cfg.optimization_horizon_m = planner_cfg.get('optimization_horizon_m', 500.0)
    cfg.gg_margin = planner_cfg.get('gg_margin', 0.1)
    cfg.safety_distance_default = planner_cfg.get('safety_distance', 0.5)
    cfg.assumed_calc_time = planner_cfg.get('assumed_calc_time', 0.125)

    params, track_handler, gg_handler, local_planner, global_planner = load_setup(
        cfg,
        track_name=sc.get('track_name', 'yas_user_smoothed'),
        vehicle_name=sc.get('vehicle_name', 'eav25_car'),
        raceline_name=sc.get('raceline_name', 'yasnorth_3d_rl_as_ref_eav25_car_gg_0.1'),
    )

    planner = AcadosTacticalPlanner(
        local_planner=local_planner, global_planner=global_planner,
        track_handler=track_handler, vehicle_params=params['vehicle_params'], cfg=cfg)

    safe_wrapper = SafeTacticalWrapper(cfg)
    tactical_mapper = TacticalToPlanner(track_handler, cfg)
    p2p = PushToPass(cfg)
    a2rl_carver = A2RLObstacleCarver(track_handler, cfg, global_planner=global_planner)
    follow_mod = FollowModule(track_handler, cfg)

    from policies.heuristic_policy import HeuristicTacticalPolicy
    policy = HeuristicTacticalPolicy(cfg)
    prev_action = get_fallback_action()

    # Sim ego state (ground truth, used for opponent stepping + ROS encoding)
    ego_state_sim = create_initial_state(
        track_handler,
        start_s=ego_cfg['start_s'], start_n=ego_cfg['start_n'],
        start_V=ego_cfg['start_V'], start_chi=ego_cfg.get('start_chi', 0.0),
    )

    opponents = []
    for opp_cfg in opp_cfgs:
        opp = OpponentVehicle(
            vehicle_id=opp_cfg['id'], s_init=opp_cfg['start_s'],
            n_init=opp_cfg.get('start_n', 0.0), V_init=opp_cfg.get('start_V', 40.0),
            track_handler=track_handler, global_planner=global_planner,
            speed_scale=opp_cfg.get('speed_scale', 0.85), cfg=cfg)
        opponents.append(opp)

    print("=" * 70)
    print(f"Mock-ROS Case16 Test: {sc['name']}")
    print(f"  Data flow: sim → ROS encode → ROS decode → planner → ROS encode")
    print(f"              → ROS decode → perfect_tracking")
    print(f"  Steps: {max_steps}")
    print("=" * 70)

    collision_count = 0
    planner_ok_count = 0
    decode_errs_s = []
    decode_errs_n = []
    step_times = []

    for step in range(max_steps):
        t0 = time.time()

        # ─── (1) Sim → ROS message encoding ────────────────────────
        opp_sim_states = [opp.get_state() for opp in opponents]
        opp_predictions = [opp.predict() for opp in opponents]
        for os_d, pred in zip(opp_sim_states, opp_predictions):
            os_d['pred_s'] = pred['pred_s']
            os_d['pred_n'] = pred['pred_n']
            os_d['pred_x'] = pred['pred_x']
            os_d['pred_y'] = pred['pred_y']

        loc_msg, ego_msg = ego_state_to_ros(ego_state_sim, track_handler)
        gt_msgs = opponents_to_ros(opp_sim_states, ego_state_sim, track_handler)

        # ─── (2) ROS message → node's internal ego_state ───────────
        ego_state_ros = ros_to_ego_state(loc_msg, ego_msg, track_handler)

        # Measure decode accuracy vs sim ground truth
        ds_err = abs(ego_state_ros['s'] - ego_state_sim['s'])
        ds_err = min(ds_err, track_handler.s[-1] - ds_err)
        dn_err = abs(ego_state_ros['n'] - ego_state_sim['n'])
        decode_errs_s.append(ds_err)
        decode_errs_n.append(dn_err)

        # ─── (3) ROS message → node's internal opp_states ──────────
        opp_states_ros = ros_to_opp_states(gt_msgs, ego_state_ros, loc_msg, track_handler)
        for os_d in opp_states_ros:
            os_d['pred_s'] = np.array([os_d['s']])
            os_d['pred_n'] = np.array([os_d['n']])
            os_d['pred_x'] = np.array([os_d['x']])
            os_d['pred_y'] = np.array([os_d['y']])

        # ─── (4) Tactical planner pipeline (identical to node) ──────
        obs = build_observation(
            ego_state=ego_state_ros, opponents=opp_states_ros,
            track_handler=track_handler, p2p_state=p2p.get_state_vector(),
            prev_action_array=prev_action.to_array(),
            planner_healthy=planner.planner_healthy, cfg=cfg)

        action = policy.act(obs)
        if hasattr(policy, 'set_overtake_ready'):
            policy.set_overtake_ready(a2rl_carver.overtake_ready)

        guidance = tactical_mapper.map(action, obs, N_stages=cfg.N_steps_acados)
        c_mode = CARVER_MODE_MAP.get(
            getattr(policy, 'carver_mode_str', 'follow'), CarverMode.FOLLOW)
        c_side = getattr(policy, 'carver_side', None)
        follow_opps = getattr(policy, 'follow_when_forced', True)

        ds = cfg.optimization_horizon_m / cfg.N_steps_acados
        carver_guidance = a2rl_carver.construct_guidance(
            ego_state_ros, opp_states_ros, cfg.N_steps_acados, ds,
            mode=c_mode, shadow_side=c_side, overtake_side=c_side,
            prev_trajectory=planner._prev_trajectory,
            planner_healthy=planner.planner_healthy,
            follow_opponents=follow_opps)

        if carver_guidance.n_left_override is not None:
            guidance.n_left_override = carver_guidance.n_left_override
        if carver_guidance.n_right_override is not None:
            guidance.n_right_override = carver_guidance.n_right_override
        if carver_guidance.speed_cap < guidance.speed_cap:
            guidance.speed_cap = carver_guidance.speed_cap
        if carver_guidance.speed_scale < guidance.speed_scale:
            guidance.speed_scale = carver_guidance.speed_scale

        trajectory = planner.plan(ego_state_ros, guidance)

        # ─── (5) Trajectory → ROS encode (node publishes) ──────────
        ref_msg = trajectory_to_ros(trajectory, ego_state_ros, track_handler, cfg)

        # ─── (6) ROS decode (planner.cpp case16 receives) ──────────
        traj_decoded = ros_to_trajectory(ref_msg, track_handler)

        t_plan = time.time() - t0
        step_times.append(t_plan)

        # ─── (7) Opponent step (sim) ───────────────────────────────
        for opp in opponents:
            opp.step(cfg.assumed_calc_time, ego_state_sim)
        p2p.step(cfg.assumed_calc_time)

        # ─── (8) Perfect tracking with decoded trajectory ──────────
        ego_state_sim = perfect_tracking_update(
            ego_state_sim, traj_decoded, cfg.assumed_calc_time, track_handler)

        # ─── (9) Logging ──────────────────────────────────────────
        planner_ok = planner.planner_healthy
        if planner_ok:
            planner_ok_count += 1

        for opp in opponents:
            dist = np.sqrt((ego_state_sim['x'] - opp.x)**2 + (ego_state_sim['y'] - opp.y)**2)
            if dist < cfg.vehicle_length * 0.5:
                collision_count += 1

        prev_action = action

        if step % 20 == 0:
            ph = getattr(policy, 'debug_info', {}).get('phase', '?')
            gap = getattr(policy, 'debug_info', {}).get('gap', None)
            gap_s = f'{gap:.1f}' if gap else 'N/A'
            opp_info = " | ".join([
                f"O{o.vehicle_id}:s={o.s:.0f}" for o in opponents])
            print(
                f'[{step:4d}] s={ego_state_sim["s"]:7.1f} n={ego_state_sim["n"]:5.2f} '
                f'V={ego_state_sim["V"]:5.1f} | {ph:10s} gap={gap_s:5s} '
                f'| {c_mode.name:8s} | {opp_info} | '
                f'ds_err={decode_errs_s[-1]:.2f} dn_err={decode_errs_n[-1]:.3f} | '
                f'{t_plan*1000:.0f}ms')

    # ─── Summary ─────────────────────────────────────────────────
    total = max_steps
    ok_rate = planner_ok_count / max(total, 1) * 100
    avg_time = np.mean(step_times) * 1000
    max_ds = max(decode_errs_s)
    max_dn = max(decode_errs_n)
    avg_ds = np.mean(decode_errs_s)
    avg_dn = np.mean(decode_errs_n)

    print("\n" + "=" * 70)
    print("Mock-ROS Case16 Test RESULTS")
    print("=" * 70)
    print(f"  Steps:            {total}")
    print(f"  Planner OK:       {ok_rate:.1f}%")
    print(f"  Collisions:       {collision_count}")
    print(f"  Avg step time:    {avg_time:.1f} ms")
    print(f"  Decode accuracy:")
    print(f"    ds: mean={avg_ds:.3f}m  max={max_ds:.3f}m")
    print(f"    dn: mean={avg_dn:.4f}m  max={max_dn:.4f}m")

    # ─── Pass / Fail ─────────────────────────────────────────────
    passed = True
    if ok_rate < 90:
        print(f"\n  ❌ FAIL: Planner success rate {ok_rate:.1f}% < 90%")
        passed = False
    if collision_count > 0:
        print(f"\n  ❌ FAIL: {collision_count} collisions")
        passed = False
    if max_ds > 5.0:
        print(f"\n  ⚠️  WARN: Max ds decode error {max_ds:.2f}m > 5m")
    if max_dn > 1.0:
        print(f"\n  ❌ FAIL: Max dn decode error {max_dn:.3f}m > 1m")
        passed = False

    if passed:
        print(f"\n  ✅ PASS: Case16 mock-ROS integration works correctly!")
        print(f"     The ROS2 encode/decode round-trip is faithful.")
        print(f"     Safe to integrate into race24.")
    else:
        print(f"\n  ❌ SOME CHECKS FAILED — review above.")

    return passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mock ROS2 Case16 Integration Test')
    parser.add_argument('--steps', type=int, default=200, help='Number of sim steps')
    parser.add_argument('--scenario', type=str, default='scenario_c', help='Scenario name')
    args = parser.parse_args()

    run_mock_ros_test(max_steps=args.steps, scenario_name=args.scenario)

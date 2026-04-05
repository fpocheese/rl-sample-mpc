#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试工具: 无可视化、定步数运行、自动分析数据
用法: python test_harness.py --scenario scenario_c --steps 1500
"""

import os, sys, time
import numpy as np

dir_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(dir_path, '..')
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, dir_path)

from sim_tactical import load_scenario, load_setup, create_initial_state, perfect_tracking_update
from config import TacticalConfig
from acados_planner import AcadosTacticalPlanner
from tactical_action import get_fallback_action
from observation import build_observation
from safe_wrapper import SafeTacticalWrapper
from planner_guidance import TacticalToPlanner
from opponent import OpponentVehicle
from p2p import PushToPass
from follow_module import FollowModule
from a2rl_obstacle_carver import A2RLObstacleCarver, CarverMode
from policies.heuristic_policy import HeuristicTacticalPolicy


def run_test(scenario_name='scenario_c', max_steps=1500, verbose=True):
    """无可视化运行仿真, 返回详细数据"""
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
    track_len = track_handler.s[-1]

    planner = AcadosTacticalPlanner(
        local_planner=local_planner, global_planner=global_planner,
        track_handler=track_handler, vehicle_params=params['vehicle_params'], cfg=cfg,
    )
    safe_wrapper = SafeTacticalWrapper(cfg)
    tactical_mapper = TacticalToPlanner(track_handler, cfg)
    p2p = PushToPass(cfg)
    a2rl_carver = A2RLObstacleCarver(track_handler, cfg, global_planner=global_planner)
    policy = HeuristicTacticalPolicy(cfg)
    follow_mod = FollowModule(track_handler, cfg)

    ego_state = create_initial_state(
        track_handler, start_s=ego_cfg['start_s'], start_n=ego_cfg['start_n'],
        start_V=ego_cfg['start_V'],
    )

    opponents = []
    for opp_cfg in opp_cfgs:
        opp = OpponentVehicle(
            vehicle_id=opp_cfg['id'], s_init=opp_cfg['start_s'],
            n_init=opp_cfg.get('start_n', 0.0), V_init=opp_cfg.get('start_V', 40.0),
            track_handler=track_handler, global_planner=global_planner,
            speed_scale=opp_cfg.get('speed_scale', 0.85), cfg=cfg,
        )
        opponents.append(opp)

    carver_mode_map = {
        'follow': CarverMode.FOLLOW, 'shadow': CarverMode.SHADOW,
        'overtake': CarverMode.OVERTAKE, 'raceline': CarverMode.RACELINE,
        'hold': CarverMode.HOLD,
    }

    prev_action = get_fallback_action()

    # === 数据记录 ===
    data = {
        'steps': [], 'ego_s': [], 'ego_n': [], 'ego_V': [],
        'phase': [], 'carver_mode': [], 'carver_side': [],
        'target_id': [], 'gap': [], 'locked_side': [],
        'collisions': [],  # (step, opp_id, dist, ego_s, ego_n, opp_n, ds)
        'overtake_events': [],  # (step, opp_id, ego_s)
        'planner_fails': [],
    }
    # 追踪超车: ego_s > opp_s 时记录
    ego_ahead_of = {}  # opp_id -> bool

    collision_count = 0
    t0 = time.time()

    for step in range(max_steps):
        # 1) Observation
        opp_predictions = [opp.predict() for opp in opponents]
        opp_states = [opp.get_state() for opp in opponents]
        for os_dict, pred in zip(opp_states, opp_predictions):
            os_dict['pred_s'] = pred['pred_s']
            os_dict['pred_n'] = pred['pred_n']
            os_dict['pred_x'] = pred['pred_x']
            os_dict['pred_y'] = pred['pred_y']

        obs = build_observation(
            ego_state=ego_state, opponents=opp_states,
            track_handler=track_handler, p2p_state=p2p.get_state_vector(),
            prev_action_array=prev_action.to_array(),
            planner_healthy=planner.planner_healthy, cfg=cfg,
        )

        # 2) Policy
        action = policy.act(obs)
        if hasattr(policy, 'set_overtake_ready'):
            policy.set_overtake_ready(a2rl_carver.overtake_ready)

        if action.p2p_trigger and p2p.available:
            p2p.activate()

        # 3) Guidance
        guidance = tactical_mapper.map(action, obs, N_stages=cfg.N_steps_acados)
        c_mode = carver_mode_map.get(getattr(policy, 'carver_mode_str', 'follow'), CarverMode.FOLLOW)
        c_side = getattr(policy, 'carver_side', None)
        ds = cfg.optimization_horizon_m / cfg.N_steps_acados

        carver_guidance = a2rl_carver.construct_guidance(
            ego_state, opp_states, cfg.N_steps_acados, ds,
            mode=c_mode, shadow_side=c_side, overtake_side=c_side,
            prev_trajectory=planner._prev_trajectory,
            planner_healthy=planner.planner_healthy,
        )
        if carver_guidance.n_left_override is not None:
            guidance.n_left_override = carver_guidance.n_left_override
        if carver_guidance.n_right_override is not None:
            guidance.n_right_override = carver_guidance.n_right_override
        if carver_guidance.speed_cap < guidance.speed_cap:
            guidance.speed_cap = carver_guidance.speed_cap
        if carver_guidance.speed_scale < guidance.speed_scale:
            guidance.speed_scale = carver_guidance.speed_scale

        # 4) Plan
        trajectory = planner.plan(ego_state, guidance)

        # 5) Move opponents & P2P
        for opp in opponents:
            opp.step(cfg.assumed_calc_time, ego_state)
        p2p.step(cfg.assumed_calc_time)

        # 6) Update ego
        ego_state = perfect_tracking_update(ego_state, trajectory, cfg.assumed_calc_time, track_handler)

        # 7) Record data
        policy_debug = getattr(policy, 'debug_info', {})
        data['steps'].append(step)
        data['ego_s'].append(ego_state['s'])
        data['ego_n'].append(ego_state['n'])
        data['ego_V'].append(ego_state['V'])
        data['phase'].append(policy_debug.get('phase', 'N/A'))
        data['carver_mode'].append(c_mode.name)
        data['carver_side'].append(c_side)
        data['target_id'].append(policy_debug.get('target_id'))
        data['gap'].append(policy_debug.get('gap'))
        data['locked_side'].append(policy_debug.get('locked_side'))

        if not planner.planner_healthy:
            # 记录 fail 详情: 可行域宽度 + 对手距离
            corridor_widths = []
            if guidance.n_left_override is not None and guidance.n_right_override is not None:
                for ci in range(min(10, len(guidance.n_left_override))):
                    cw = guidance.n_left_override[ci] - guidance.n_right_override[ci]
                    corridor_widths.append(cw)
            min_cw = min(corridor_widths) if corridor_widths else -1
            opp_dists = []
            for opp in opponents:
                ds_o = opp.s - ego_state['s']
                opp_dists.append((opp.vehicle_id, ds_o, opp.n))
            data['planner_fails'].append({
                'step': step, 's': ego_state['s'], 'n': ego_state['n'],
                'V': ego_state['V'], 'chi': ego_state.get('chi', 0),
                'mode': c_mode.name, 'side': c_side,
                'min_corridor': min_cw,
                'opp_dists': opp_dists,
            })

        # 8) Collision check
        for opp in opponents:
            dist = np.sqrt((ego_state['x'] - opp.x)**2 + (ego_state['y'] - opp.y)**2)
            if dist < cfg.vehicle_length * 0.5:
                collision_count += 1
                ds_col = ego_state['s'] - opp.s
                data['collisions'].append((step, opp.vehicle_id, dist,
                    ego_state['s'], ego_state['n'], opp.n, ds_col,
                    policy_debug.get('phase', 'N/A'), c_mode.name))

        # 9) Track overtake events
        for opp in opponents:
            ds_opp = ego_state['s'] - opp.s
            if ds_opp > track_len / 2: ds_opp -= track_len
            elif ds_opp < -track_len / 2: ds_opp += track_len
            was_ahead = ego_ahead_of.get(opp.vehicle_id, False)
            now_ahead = ds_opp > 5.0
            if now_ahead and not was_ahead:
                data['overtake_events'].append((step, opp.vehicle_id, ego_state['s']))
            ego_ahead_of[opp.vehicle_id] = now_ahead

        # 10) Print periodic
        if verbose and step % 100 == 0:
            gap_str = f"{policy_debug.get('gap', 0):5.1f}" if policy_debug.get('gap') is not None else "  N/A"
            print(f"[{step:4d}] s={ego_state['s']:7.1f} n={ego_state['n']:5.2f} "
                  f"V={ego_state['V']:5.1f} | {policy_debug.get('phase','N/A'):10s} "
                  f"gap={gap_str} | {c_mode.name:8s} {c_side or '':5s} | col={collision_count}")

        prev_action = action

        if ego_state['s'] > sc.get('s_end', 1e6):
            break

    elapsed = time.time() - t0
    return data, collision_count, elapsed


def analyze(data, collision_count, elapsed, max_steps):
    """分析测试数据, 打印报告"""
    n = len(data['steps'])
    print("\n" + "=" * 70)
    print("测试报告")
    print("=" * 70)
    print(f"总步数: {n}, 运行时间: {elapsed:.1f}s ({elapsed/max(n,1)*1000:.1f}ms/step)")
    print(f"碰撞总数: {collision_count}")
    print(f"Planner失败次数: {len(data['planner_fails'])}")

    # Planner fail 诊断
    if data['planner_fails']:
        print("\n--- Planner失败诊断 ---")
        # 按s位置分桶统计
        from collections import Counter, defaultdict
        fail_s_buckets = defaultdict(list)
        for f in data['planner_fails']:
            bucket = int(f['s'] / 100) * 100
            fail_s_buckets[bucket].append(f)
        print("  按位置分布:")
        for bucket in sorted(fail_s_buckets.keys()):
            fails = fail_s_buckets[bucket]
            modes = Counter(f['mode'] for f in fails)
            min_cw = min(f['min_corridor'] for f in fails)
            avg_v = np.mean([f['V'] for f in fails])
            print(f"    s={bucket:4d}~{bucket+100}: {len(fails):3d}次 "
                  f"min_corridor={min_cw:.2f}m avg_V={avg_v:.1f} modes={dict(modes)}")
        # 前5个fail详情
        print("  前5个fail详情:")
        for f in data['planner_fails'][:5]:
            opp_str = " ".join(f"Opp{oid}(ds={ds:.1f},n={n:.1f})" for oid, ds, n in f['opp_dists'])
            print(f"    Step {f['step']}: s={f['s']:.0f} n={f['n']:.2f} V={f['V']:.1f} "
                  f"chi={f['chi']:.3f} {f['mode']} {f['side'] or ''} "
                  f"corridor={f['min_corridor']:.2f}m | {opp_str}")
        # 代表性采样: 每50个fail取1个
        if len(data['planner_fails']) > 10:
            print("  采样详情(每50个):")
            for idx in range(0, len(data['planner_fails']), 50):
                f = data['planner_fails'][idx]
                opp_str = " ".join(f"Opp{oid}(ds={ds:.1f},n={n:.1f})" for oid, ds, n in f['opp_dists'])
                print(f"    [{idx}] Step {f['step']}: s={f['s']:.0f} n={f['n']:.2f} V={f['V']:.1f} "
                      f"chi={f['chi']:.3f} {f['mode']} corridor={f['min_corridor']:.2f}m | {opp_str}")

    # 碰撞详情
    if data['collisions']:
        print("\n--- 碰撞详情 ---")
        for col in data['collisions']:
            step, opp_id, dist, ego_s, ego_n, opp_n, ds, phase, cmode = col
            print(f"  Step {step}: Opp{opp_id} dist={dist:.2f}m ego_s={ego_s:.0f} "
                  f"ego_n={ego_n:.2f} opp_n={opp_n:.2f} ds={ds:.1f} "
                  f"[{phase}/{cmode}]")

    # 超车事件
    print(f"\n--- 超车事件 ({len(data['overtake_events'])}) ---")
    for evt in data['overtake_events']:
        step, opp_id, ego_s = evt
        print(f"  Step {step}: 超过 Opp{opp_id} at s={ego_s:.0f}")

    # 各阶段统计
    from collections import Counter
    phase_counts = Counter(data['phase'])
    print("\n--- 阶段分布 ---")
    for phase, cnt in phase_counts.most_common():
        print(f"  {phase:12s}: {cnt:5d} ({cnt/max(n,1)*100:.1f}%)")

    carver_counts = Counter(data['carver_mode'])
    print("\n--- Carver模式分布 ---")
    for mode, cnt in carver_counts.most_common():
        print(f"  {mode:12s}: {cnt:5d} ({cnt/max(n,1)*100:.1f}%)")

    # 速度统计
    V = np.array(data['ego_V'])
    print(f"\n--- 速度统计 ---")
    print(f"  平均: {np.mean(V):.1f} m/s, 最大: {np.max(V):.1f}, 最小: {np.min(V):.1f}")

    # Gap统计 (非None值)
    gaps = [g for g in data['gap'] if g is not None]
    if gaps:
        gaps_arr = np.array(gaps)
        print(f"\n--- Gap统计 ---")
        print(f"  平均: {np.mean(gaps_arr):.1f}m, 最小: {np.min(gaps_arr):.1f}m, 最大: {np.max(gaps_arr):.1f}m")

    # 查找 OT-HOLD 振荡
    phases = data['phase']
    oscillations = 0
    for i in range(2, n):
        if phases[i] == 'OVERTAKE' and phases[i-1] == 'HOLD' and i > 5:
            # 检查前5步是否有OVERTAKE
            recent = phases[max(0, i-10):i]
            if 'OVERTAKE' in recent:
                oscillations += 1
    if oscillations > 0:
        print(f"\n⚠️  检测到 OT-HOLD 振荡: {oscillations} 次")

    # 找到最小gap时刻的phase
    if gaps:
        min_gap_idx = np.argmin(gaps_arr)
        # 找到对应的data index
        gap_indices = [i for i, g in enumerate(data['gap']) if g is not None]
        if min_gap_idx < len(gap_indices):
            idx = gap_indices[min_gap_idx]
            print(f"\n--- 最近距离时刻 ---")
            print(f"  Step {data['steps'][idx]}: gap={gaps_arr[min_gap_idx]:.1f}m "
                  f"phase={data['phase'][idx]} carver={data['carver_mode'][idx]} "
                  f"s={data['ego_s'][idx]:.0f} V={data['ego_V'][idx]:.1f}")

    print("\n" + "=" * 70)
    return collision_count == 0 and len(data['overtake_events']) >= 2


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='scenario_c')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()

    data, col, elapsed = run_test(args.scenario, args.steps, not args.quiet)
    success = analyze(data, col, elapsed, args.steps)
    if success:
        print("✅ 通过: 零碰撞 + 至少超过2个对手")
    else:
        print("❌ 未通过: 需要继续优化")

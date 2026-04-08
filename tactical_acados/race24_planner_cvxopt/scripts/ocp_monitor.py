#!/usr/bin/env python3
"""
OCP Planner GG Monitor Node
============================
Subscribes to:
  - /flyeagle/a2rl/vn/ins : VectornavIns (actual ax, ay, speed)
  - /flyeagle/a2rl/planner/trajectory : ReferencePath (OCP planned speed)
  - /flyeagle/a2rl/planner/reference_s_distance : Float32MultiArray (GG limits at current speed)

Publishes to foxglove (Float32MultiArray on /ocp_monitor/gg_status):
  [0] V_actual      : actual vehicle speed (m/s)
  [1] ax_actual      : actual longitudinal accel (m/s²)
  [2] ay_actual      : actual lateral accel (m/s²)
  [3] An_limit       : lateral accel limit at current V (from s_distance[7])
  [4] Aw_limit       : braking limit at current V (from s_distance[10], abs)
  [5] Ae0_limit      : engine accel limit at current V (from s_distance[9])
  [6] gg_ratio       : (|ax|/ax_lim) + (|ay|/ay_lim), diamond ratio, >1 = violation
  [7] V_planned_0    : first planned speed from OCP trajectory
  [8] V_planned_5    : 5th planned speed from OCP trajectory
  [9] V_raceline     : raceline reference speed at current s (from s_distance data, approximate)
  [10] ay_max_used    : ay limit used for gg_ratio computation
  [11] ax_lim_used    : ax limit used for gg_ratio computation (Aw or Ae0 depending on sign)
  [12] gg_violation   : 1.0 if gg_ratio > 1.0, else 0.0

Also prints colorized terminal output every cycle.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float32MultiArray
from a2rl_bs_msgs.msg import VectornavIns, ReferencePath
import math
import numpy as np
import os
import csv

class OCPMonitor(Node):
    def __init__(self):
        super().__init__('ocp_monitor')
        
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        
        # Subscribers
        self.ins_sub = self.create_subscription(
            VectornavIns, '/flyeagle/a2rl/vn/ins',
            self.ins_cb, qos_best_effort)
        
        self.traj_sub = self.create_subscription(
            ReferencePath, '/flyeagle/a2rl/planner/trajectory',
            self.traj_cb, qos_reliable)
        
        self.sdist_sub = self.create_subscription(
            Float32MultiArray, '/flyeagle/a2rl/planner/reference_s_distance',
            self.sdist_cb, qos_reliable)
        
        # Publisher
        self.pub = self.create_publisher(Float32MultiArray, '/ocp_monitor/gg_status', 10)
        
        # Load CarData2025 for our own GG lookup
        self.load_cardata()
        
        # State
        self.ins_msg = None
        self.traj_msg = None
        self.sdist_data = None
        
        # Timer: 10 Hz monitor
        self.timer = self.create_timer(0.1, self.monitor_cb)
        
        self.violation_count = 0
        self.total_count = 0
        
        self.get_logger().info('\033[1;32m[OCP Monitor] Started. Publishing to /ocp_monitor/gg_status\033[0m')
    
    def load_cardata(self):
        """Load CarData2025.csv for direct GG limit lookup by velocity."""
        cardata_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', 
            'config', 'tracks', 'North_Line', 'CarData2025.csv')
        
        self.cd_V = []
        self.cd_Aw = []  # longitudinal braking limit
        self.cd_An = []  # lateral limit  
        self.cd_Ae0 = [] # engine accel limit
        
        try:
            with open(cardata_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.cd_V.append(float(row['V']))
                    self.cd_Aw.append(float(row['Aw']))
                    self.cd_An.append(float(row['An']))
                    self.cd_Ae0.append(float(row['Ae_on']))
            self.cd_V = np.array(self.cd_V)
            self.cd_Aw = np.array(self.cd_Aw)
            self.cd_An = np.array(self.cd_An)
            self.cd_Ae0 = np.array(self.cd_Ae0)
            self.get_logger().info(f'[OCP Monitor] Loaded CarData2025: {len(self.cd_V)} points, V=[{self.cd_V[0]:.0f}, {self.cd_V[-1]:.0f}] m/s')
        except Exception as e:
            self.get_logger().error(f'[OCP Monitor] Failed to load CarData2025: {e}')
            self.cd_V = np.array([0, 100])
            self.cd_Aw = np.array([20, 20])
            self.cd_An = np.array([15, 15])
            self.cd_Ae0 = np.array([10, 10])
    
    def interp_gg(self, V):
        """Interpolate GG limits at given speed."""
        An = float(np.interp(V, self.cd_V, self.cd_An))   # lateral (ay_max)
        Aw = float(np.interp(V, self.cd_V, self.cd_Aw))   # braking (ax_min)
        Ae0 = float(np.interp(V, self.cd_V, self.cd_Ae0)) # engine (ax_max)
        return An, Aw, Ae0
    
    def ins_cb(self, msg):
        self.ins_msg = msg
    
    def traj_cb(self, msg):
        self.traj_msg = msg
    
    def sdist_cb(self, msg):
        self.sdist_data = msg.data
    
    def monitor_cb(self):
        if self.ins_msg is None:
            return
        
        # ---- Actual state from INS ----
        # acceleration_ins: x = longitudinal (body), y = lateral (body)
        ax_actual = self.ins_msg.acceleration_ins.x
        ay_actual = self.ins_msg.acceleration_ins.y
        
        # Speed from INS body frame
        vx = self.ins_msg.velocity_body_ins.x
        vy = self.ins_msg.velocity_body_ins.y
        V_actual = math.sqrt(vx * vx + vy * vy)
        
        # ---- GG limits from our CarData lookup ----
        An_limit, Aw_limit, Ae0_limit = self.interp_gg(V_actual)
        
        # ---- GG diamond ratio (exponent=1.0) ----
        # ax_lim = Ae0 if accelerating, Aw if braking
        ax_lim = Ae0_limit if ax_actual >= 0 else Aw_limit
        ay_lim = An_limit
        
        gg_ratio = 0.0
        if ax_lim > 0.01 and ay_lim > 0.01:
            gg_ratio = abs(ax_actual) / ax_lim + abs(ay_actual) / ay_lim
        
        gg_violation = 1.0 if gg_ratio > 1.0 else 0.0
        
        # ---- Planned speeds from OCP trajectory ----
        V_planned_0 = 0.0
        V_planned_5 = 0.0
        if self.traj_msg is not None and len(self.traj_msg.path) > 0:
            V_planned_0 = self.traj_msg.path[0].velocity_linear.x
            if len(self.traj_msg.path) > 5:
                V_planned_5 = self.traj_msg.path[5].velocity_linear.x
        
        # ---- s_distance data (from planner, index 7=An, 9=Ae0, 10=-Aw) ----
        An_from_planner = 0.0
        Ae0_from_planner = 0.0
        Aw_from_planner = 0.0
        V_raceline_approx = 0.0
        if self.sdist_data is not None and len(self.sdist_data) > 10:
            An_from_planner = self.sdist_data[7]   # lateral limit
            Ae0_from_planner = self.sdist_data[9]  # engine accel limit  
            Aw_from_planner = abs(self.sdist_data[10])  # braking limit (stored as negative)
        
        # ---- Publish ----
        out = Float32MultiArray()
        out.data = [
            float(V_actual),       # 0
            float(ax_actual),      # 1
            float(ay_actual),      # 2
            float(An_limit),       # 3: lateral limit (ay_max)
            float(Aw_limit),       # 4: braking limit
            float(Ae0_limit),      # 5: engine accel limit
            float(gg_ratio),       # 6: diamond GG ratio (>1 = violation)
            float(V_planned_0),    # 7: first planned speed
            float(V_planned_5),    # 8: 5th planned speed
            float(V_raceline_approx),  # 9: raceline speed (not yet populated)
            float(ay_lim),         # 10: ay limit used
            float(ax_lim),         # 11: ax limit used
            float(gg_violation),   # 12: 1 if violated
        ]
        self.pub.publish(out)
        
        # ---- Terminal output ----
        self.total_count += 1
        if gg_violation > 0.5:
            self.violation_count += 1
        
        # Color coding
        if gg_ratio > 1.2:
            color = '\033[1;31m'  # bright red - severe violation
            tag = '!!! OVER !!!'
        elif gg_ratio > 1.0:
            color = '\033[1;33m'  # yellow - marginal violation
            tag = '! MARGINAL !'
        elif gg_ratio > 0.8:
            color = '\033[1;36m'  # cyan - approaching limit
            tag = '  CLOSE   '
        else:
            color = '\033[0;32m'  # green - safe
            tag = '    OK    '
        reset = '\033[0m'
        
        # Print every 5th cycle (2 Hz at 10 Hz timer)
        if self.total_count % 5 == 0:
            viol_pct = 100.0 * self.violation_count / max(self.total_count, 1)
            print(f'{color}[GG] V={V_actual:5.1f} '
                  f'ax={ax_actual:+6.1f}({ax_lim:5.1f}) '
                  f'ay={ay_actual:+6.1f}({ay_lim:5.1f}) '
                  f'ratio={gg_ratio:.2f} [{tag}] '
                  f'Vplan={V_planned_0:5.1f}/{V_planned_5:5.1f} '
                  f'viol={viol_pct:.0f}%{reset}')


def main(args=None):
    rclpy.init(args=args)
    node = OCPMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

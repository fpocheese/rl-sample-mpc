"""
Lightweight tactical visualizer for the ACADOS-only pipeline.

Simplified version of the original Visualizer that works with the
tactical trajectory format (dict with t, s, n, V, x, y, z, etc.).
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.patches import Rectangle
mpl.rcParams['lines.linewidth'] = 2


class TacticalVisualizer:
    """Real-time visualization for the tactical ACADOS simulation."""

    def __init__(self, track_handler, gg_handler, params,
                 zoom_on_ego=True, zoom_margin=30.0, n_opponents=0):
        self.track_handler = track_handler
        self.gg_handler = gg_handler
        self.params = params
        self.zoom_on_ego = zoom_on_ego
        self.zoom_margin = zoom_margin

        plt.ion()
        self.fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1],
                               width_ratios=[1.5, 1], figure=self.fig)

        # Track subplot
        self.ax_track = plt.subplot(gs[:, 0])
        self.ax_track.set_xlabel('x [m]')
        self.ax_track.set_ylabel('y [m]')
        self.ax_track.set_aspect('equal', adjustable='datalim')
        self.ax_track.grid(True, alpha=0.3)

        left, right = track_handler.get_track_bounds()
        self.ax_track.plot(left[0], left[1], 'k', linewidth=1.5)
        self.ax_track.plot(right[0], right[1], 'k', linewidth=1.5)

        self.traj_line, = self.ax_track.plot([], [], 'b-', linewidth=2,
                                              label='Trajectory')
        veh = params['vehicle_params']
        self.ego_rect = Rectangle((0, 0), veh['total_length'], veh['total_width'],
                                  angle=0, color='dodgerblue', alpha=0.8)
        self.ax_track.add_patch(self.ego_rect)

        # Opponent visualization
        self.opp_lines = []
        self.opp_rects = []
        for i in range(n_opponents):
            ln, = self.ax_track.plot([], [], 'r-', linewidth=1.5,
                                      label='Opponent' if i == 0 else None)
            rect = Rectangle((0, 0), veh['total_length'], veh['total_width'],
                              angle=0, color='crimson', alpha=0.7)
            self.ax_track.add_patch(rect)
            self.opp_lines.append(ln)
            self.opp_rects.append(rect)

        self.ax_track.legend(loc='upper right', fontsize=8)

        # Velocity subplot
        self.ax_vel = plt.subplot(gs[0, 1])
        self.ax_vel.set_xlabel('t [s]')
        self.ax_vel.set_ylabel('V [m/s]')
        self.ax_vel.grid(True, alpha=0.3)
        self.vel_line, = self.ax_vel.plot([], [], 'b-', linewidth=2)

        # Lateral offset subplot
        self.ax_n = plt.subplot(gs[1, 1])
        self.ax_n.set_xlabel('t [s]')
        self.ax_n.set_ylabel('n [m]')
        self.ax_n.grid(True, alpha=0.3)
        self.n_line, = self.ax_n.plot([], [], 'b-', linewidth=2)

        # Info text box
        self.ax_info = plt.subplot(gs[2, 1])
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(0.05, 0.95, '', transform=self.ax_info.transAxes,
                                           fontsize=9, verticalalignment='top',
                                           fontfamily='monospace')

        self.fig.tight_layout()

    def update(self, state, trajectory, opponents=None, tactical_info=None):
        """Update all plots with current state and trajectory."""
        # Trajectory on track
        self.traj_line.set_xdata(trajectory['x'])
        self.traj_line.set_ydata(trajectory['y'])

        # Ego rectangle
        heading = self.track_handler.calc_2d_heading_from_chi(state['chi'], state['s'])
        veh = self.params['vehicle_params']
        self.ego_rect.set_x(state['x'] - veh['total_length']/2 * np.cos(heading)
                            + veh['total_width']/2 * np.sin(heading))
        self.ego_rect.set_y(state['y'] - veh['total_length']/2 * np.sin(heading)
                            - veh['total_width']/2 * np.cos(heading))
        self.ego_rect.set_angle(np.rad2deg(heading))

        # Zoom
        if self.zoom_on_ego:
            self.ax_track.set_xlim([min(trajectory['x']) - self.zoom_margin,
                                    max(trajectory['x']) + self.zoom_margin])
            self.ax_track.set_ylim([min(trajectory['y']) - self.zoom_margin,
                                    max(trajectory['y']) + self.zoom_margin])

        # Opponents
        if opponents is not None:
            for i, opp in enumerate(opponents):
                if i < len(self.opp_lines):
                    self.opp_lines[i].set_xdata(opp.get('pred_x', []))
                    self.opp_lines[i].set_ydata(opp.get('pred_y', []))
                    if 'x' in opp and 'y' in opp:
                        opp_heading = self.track_handler.calc_2d_heading_from_chi(
                            opp.get('chi', 0.0), opp.get('s', 0.0))
                        self.opp_rects[i].set_x(
                            opp['x'] - veh['total_length']/2 * np.cos(opp_heading)
                            + veh['total_width']/2 * np.sin(opp_heading))
                        self.opp_rects[i].set_y(
                            opp['y'] - veh['total_length']/2 * np.sin(opp_heading)
                            - veh['total_width']/2 * np.cos(opp_heading))
                        self.opp_rects[i].set_angle(np.rad2deg(opp_heading))

        # Velocity
        self.vel_line.set_xdata(trajectory['t'])
        self.vel_line.set_ydata(trajectory['V'])
        self.ax_vel.set_xlim([0, trajectory['t'][-1]])
        self.ax_vel.set_ylim([0, max(trajectory['V']) * 1.2 + 1])

        # Lateral offset
        self.n_line.set_xdata(trajectory['t'])
        self.n_line.set_ydata(trajectory['n'])
        self.ax_n.set_xlim([0, trajectory['t'][-1]])
        n_range = max(abs(min(trajectory['n'])), abs(max(trajectory['n']))) + 2
        self.ax_n.set_ylim([-n_range, n_range])

        # Info box
        info = (f"s  = {state['s']:8.2f} m\n"
                f"n  = {state['n']:6.3f} m\n"
                f"V  = {state['V']:6.2f} m/s\n"
                f"chi= {np.rad2deg(state['chi']):6.2f} deg\n"
                f"ax = {state['ax']:6.2f} m/s²\n"
                f"ay = {state['ay']:6.2f} m/s²")
        if tactical_info:
            info += f"\n--- Tactical ---\n{tactical_info}"
        self.info_text.set_text(info)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

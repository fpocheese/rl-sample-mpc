#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_common.py  v3  —  Shared plotting & utility module for all paper figures.

IEEE top-journal quality (Automatica / TRO / RAL grade):
  • Nature / Science colour palette, colourblind-safe
  • True N-step corridor from decision layer (not 1-point approximation)
  • Oriented vehicle rectangles with heading arrows in zoomed insets
  • Corridor evolution envelope on trajectory overview
  • Decision-layer annotations (phase, carver mode, locked side)
  • Smoothed opponent trajectory (no jumps)
  • Time-series: 4 game-theoretic sub-plots, no speed plot
  • Sub-figure labels (a)(b)… below each panel, no titles on top

Shared by:  paper_turn678.py, paper_turn234.py, paper_turn5.py
"""

import os, sys, copy
import numpy as np

# ======================================================================
# 1. IEEE RC-params  (Automatica / TRO level)
# ======================================================================
def setup_ieee():
    """Apply IEEE-quality matplotlib rc params."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rcParams
    rcParams.update({
        # --- font ---
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 8,
        'mathtext.fontset': 'stix',
        # --- axes ---
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'axes.linewidth': 0.5,
        'axes.grid': False,              # no grid on trajectory panels
        'axes.unicode_minus': False,
        'axes.facecolor': '#FAFAFA',     # very faint warm grey bg
        # --- ticks ---
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.width': 0.4,
        'ytick.major.width': 0.4,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        # --- legend ---
        'legend.fontsize': 6,
        'legend.framealpha': 0.92,
        'legend.edgecolor': '#cccccc',
        'legend.handlelength': 1.2,
        'legend.handletextpad': 0.3,
        'legend.columnspacing': 0.6,
        'legend.borderpad': 0.25,
        'legend.labelspacing': 0.3,
        # --- lines ---
        'lines.linewidth': 0.8,
        'lines.markersize': 3,
        # --- grid ---
        'grid.alpha': 0.15,
        'grid.linewidth': 0.25,
        'grid.linestyle': ':',
        # --- figure ---
        'figure.dpi': 300,
        'savefig.dpi': 600,              # high-res for print
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


# ======================================================================
# 2. Scientific colour palette  (Nature / IEEE TRO-friendly)
# ======================================================================
# Primary vehicles
CLR_EGO        = '#1B66AB'   # muted blue
CLR_OPP        = '#C0392B'   # muted red
CLR_RACELINE   = '#7F8C8D'   # grey

# Track
CLR_TRACK_EDGE = '#2C3E50'   # dark charcoal
CLR_TRACK_CL   = '#D5D8DC'   # light grey  (centre-line)
CLR_TRACK_FILL = '#F0F0F0'   # very faint track surface

# Corridor (feasible region) — dual-tone
CLR_CORRIDOR        = '#27AE60'   # green edge
CLR_CORRIDOR_FILL   = '#2ECC71'   # slightly brighter fill
CLR_CORRIDOR_FAINT  = '#82E0AA'   # faint envelope on overview

# Decision phase background colours (muted, low saturation)
PHASE_COLORS = {
    'RACELINE':  '#E8EAF6',   # lavender
    'SHADOW':    '#FFF8E1',   # pale amber
    'OVERTAKE':  '#E8F5E9',   # pale green
    'HOLD':      '#FFEBEE',   # pale red
}

# Zoom-box border colours (distinguishable, colourblind-safe)
ZOOM_COLORS = ['#E67E22', '#2980B9', '#27AE60']   # orange, blue, green

# Vehicle fill
CLR_EGO_FILL   = '#AED6F1'   # light blue
CLR_OPP_FILL   = '#F5B7B1'   # light red

# Overtake star
CLR_OT_STAR    = '#F39C12'


# ======================================================================
# 3. Smoothing utilities
# ======================================================================
def smooth_trajectory_xy(x_arr, y_arr, window=7):
    """
    Smooth XY trajectory to remove discrete lateral jumps.
    Detect jumps > 3× median step distance, interpolate, then filter.
    """
    x = np.array(x_arr, dtype=float)
    y = np.array(y_arr, dtype=float)
    if len(x) < window:
        return x, y
    dx = np.diff(x)
    dy = np.diff(y)
    step_d = np.sqrt(dx**2 + dy**2)
    med_d = max(np.median(step_d), 1e-6)
    for i in range(1, len(x)-1):
        if step_d[i-1] > 3.0 * med_d:
            x[i] = 0.5 * (x[i-1] + x[i+1])
            y[i] = 0.5 * (y[i-1] + y[i+1])
    from scipy.ndimage import uniform_filter1d
    x_s = uniform_filter1d(x, size=window, mode='nearest')
    y_s = uniform_filter1d(y, size=window, mode='nearest')
    return x_s, y_s


def smooth_n_trajectory(n_arr, window=7):
    """Smooth lateral n sequence."""
    n = np.array(n_arr, dtype=float)
    if len(n) < window:
        return n
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(n, size=window, mode='nearest')


# ======================================================================
# 4. Vehicle rectangle + heading arrow
# ======================================================================
def draw_vehicle_rect(ax, x, y, heading, length=4.9, width=1.93,
                      facecolor='#AED6F1', edgecolor='#1B66AB',
                      linewidth=0.5, alpha=0.88, zorder=10, label=None,
                      draw_arrow=True, arrow_len=3.5):
    """
    Draw an oriented vehicle rectangle with optional heading arrow.
    """
    import matplotlib.patches as mpatches
    import matplotlib.transforms as mtransforms

    rect = mpatches.FancyBboxPatch(
        (-length/2, -width/2), length, width,
        boxstyle="round,pad=0.12",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, alpha=alpha, zorder=zorder,
        label=label,
    )
    t = mtransforms.Affine2D().rotate(heading).translate(x, y) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)

    # Heading arrow (shows driving direction)
    if draw_arrow:
        dx = arrow_len * np.cos(heading)
        dy = arrow_len * np.sin(heading)
        ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                    arrowprops=dict(arrowstyle='->', color=edgecolor,
                                    lw=0.6, shrinkA=0, shrinkB=0),
                    zorder=zorder + 1)
    return rect


def estimate_heading(x_arr, y_arr, idx, dt=0.125):
    """Estimate heading from trajectory using central difference."""
    N = len(x_arr)
    if N < 2:
        return 0.0
    i0 = max(0, idx - 3)
    i1 = min(N - 1, idx + 3)
    if i1 == i0:
        return 0.0
    dx = x_arr[i1] - x_arr[i0]
    dy = y_arr[i1] - y_arr[i0]
    return float(np.arctan2(dy, dx))


# ======================================================================
# 5. TRUE N-step corridor drawing (from full recorded sequence)
# ======================================================================
def draw_true_corridor(ax, rec, track_handler, step_idx,
                       alpha_fill=0.18, alpha_line=0.55,
                       max_nodes=80, linewidth=0.6):
    """
    Draw the REAL N-step corridor (feasible region) at a specific sim step.
    Uses corridor_s_full / corridor_left_full / corridor_right_full
    which are the complete 150-node arrays from the carver.

    Falls back to the scalar corridor_left/corridor_right if full data
    is not available.
    """
    track_len = track_handler.s[-1]

    # ---- Try full N-step corridor first ----
    if ('corridor_s_full' in rec
            and step_idx < len(rec['corridor_s_full'])
            and rec['corridor_s_full'][step_idx] is not None):
        s_full = np.array(rec['corridor_s_full'][step_idx])
        nl_full = np.array(rec['corridor_left_full'][step_idx])
        nr_full = np.array(rec['corridor_right_full'][step_idx])

        # Subsample for performance (every other node)
        step = max(1, len(s_full) // max_nodes)
        idxs = np.arange(0, len(s_full), step)
        s_sub = s_full[idxs] % track_len
        nl_sub = nl_full[idxs]
        nr_sub = nr_full[idxs]

        try:
            xyz_left = track_handler.sn2cartesian(s_sub, nl_sub)
            xyz_right = track_handler.sn2cartesian(s_sub, nr_sub)

            # Filled polygon
            poly_x = np.concatenate([xyz_left[:, 0], xyz_right[::-1, 0]])
            poly_y = np.concatenate([xyz_left[:, 1], xyz_right[::-1, 1]])
            ax.fill(poly_x, poly_y,
                    color=CLR_CORRIDOR_FILL, alpha=alpha_fill, zorder=1,
                    label=None)
            # Boundary lines (dashed for elegance)
            ax.plot(xyz_left[:, 0], xyz_left[:, 1], '-',
                    color=CLR_CORRIDOR, linewidth=linewidth,
                    alpha=alpha_line, zorder=2)
            ax.plot(xyz_right[:, 0], xyz_right[:, 1], '-',
                    color=CLR_CORRIDOR, linewidth=linewidth,
                    alpha=alpha_line, zorder=2)
            return  # success
        except Exception:
            pass  # fall through to scalar fallback

    # ---- Scalar fallback (v2 compatible) ----
    if step_idx < len(rec.get('corridor_left', [])):
        n_left_val = rec['corridor_left'][step_idx]
        n_right_val = rec['corridor_right'][step_idx]
        ego_s = rec['ego_s'][step_idx]
        s_band = np.linspace(ego_s - 15, ego_s + 25, 60) % track_len
        try:
            xyz_l = track_handler.sn2cartesian(s_band, np.full_like(s_band, n_left_val))
            xyz_r = track_handler.sn2cartesian(s_band, np.full_like(s_band, n_right_val))
            ax.fill(np.concatenate([xyz_l[:, 0], xyz_r[::-1, 0]]),
                    np.concatenate([xyz_l[:, 1], xyz_r[::-1, 1]]),
                    color=CLR_CORRIDOR_FILL, alpha=alpha_fill * 0.6, zorder=1)
        except Exception:
            pass


def draw_corridor_envelope_overview(ax, rec, track_handler,
                                    step_every=4, alpha_fill=0.06,
                                    max_ahead=50):
    """
    Draw a faint corridor envelope on the overview panel by overlaying
    corridor polygons from multiple time steps.  Shows how the feasible
    region evolves over the whole scenario.
    """
    track_len = track_handler.s[-1]
    if 'corridor_s_full' not in rec:
        return

    N = len(rec['corridor_s_full'])
    for si in range(0, N, step_every):
        if rec['corridor_s_full'][si] is None:
            continue
        s_full = np.array(rec['corridor_s_full'][si])
        nl = np.array(rec['corridor_left_full'][si])
        nr = np.array(rec['corridor_right_full'][si])
        # Only use first max_ahead nodes (near ego)
        n_use = min(len(s_full), max_ahead)
        s_sub = s_full[:n_use] % track_len
        try:
            xyz_l = track_handler.sn2cartesian(s_sub, nl[:n_use])
            xyz_r = track_handler.sn2cartesian(s_sub, nr[:n_use])
            px = np.concatenate([xyz_l[:, 0], xyz_r[::-1, 0]])
            py = np.concatenate([xyz_l[:, 1], xyz_r[::-1, 1]])
            ax.fill(px, py, color=CLR_CORRIDOR_FAINT,
                    alpha=alpha_fill, zorder=0, linewidth=0)
        except Exception:
            continue


# ======================================================================
# 6. Decision annotation (improved: icons + concise text)
# ======================================================================
def annotate_decision(ax, rec, step_idx, x_pos, y_pos,
                      fontsize=5.5, show_phase=True, show_mode=True,
                      show_side=True, offset_pts=(0, -8)):
    """
    Text annotation showing tactical decision state.
    Uses a clean rounded box with colour-coded border.
    """
    parts = []
    phase_str = ''
    if show_phase and step_idx < len(rec['phase']):
        phase_str = rec['phase'][step_idx]
        parts.append(phase_str)
    if show_mode and step_idx < len(rec['carver_mode']):
        parts.append(rec['carver_mode'][step_idx])
    if show_side and step_idx < len(rec.get('locked_side', [])):
        side = rec['locked_side'][step_idx]
        if side and side != 'None':
            parts.append(f'→{side}')

    text = ' · '.join(parts)
    if not text:
        return

    # Colour-code the box border by phase
    border_color = {
        'RACELINE': '#7986CB', 'SHADOW': '#FFB74D',
        'OVERTAKE': '#66BB6A', 'HOLD': '#EF5350',
    }.get(phase_str, '#999999')

    bbox = dict(boxstyle='round,pad=0.15', facecolor='white',
                edgecolor=border_color, alpha=0.92, linewidth=0.5)
    ax.annotate(text, xy=(x_pos, y_pos),
                fontsize=fontsize, ha='center', va='top',
                bbox=bbox, zorder=15,
                xytext=offset_pts, textcoords='offset points')


# ======================================================================
# 7. Trajectory figure  (overview + 3 zoomed insets) — v3
# ======================================================================
def plot_trajectory(rec, events, track_handler, save_path,
                    s_plot_min=2390, s_plot_max=2860,
                    zoom_pad=20, zoom_moments=None,
                    corridor_full_seq=True):
    """
    IEEE wide horizontal figure:  [overview | zoom1 | zoom2 | zoom3]

    v3 improvements over v2:
      - True N-step corridor in zoom insets (not scalar approximation)
      - Faint corridor envelope on overview
      - Vehicle heading arrows
      - Track surface fill
      - Phase-coloured annotation boxes
      - Improved legend placement
    """
    setup_ieee()
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    track_len = track_handler.s[-1]
    s_arr = track_handler.s
    mask_t = (s_arr >= s_plot_min) & (s_arr <= s_plot_max)
    s_reg = s_arr[mask_t]

    # Track boundaries
    xyz_l = track_handler.sn2cartesian(s_reg, track_handler.w_tr_left[mask_t])
    xyz_r = track_handler.sn2cartesian(s_reg, track_handler.w_tr_right[mask_t])
    xyz_c = track_handler.sn2cartesian(s_reg, np.zeros_like(s_reg))

    ego_x = np.array(rec['ego_x']); ego_y = np.array(rec['ego_y'])
    ego_s = np.array(rec['ego_s']); t_arr = np.array(rec['time'])
    N = len(ego_x)

    opp_ids = sorted(set(
        int(k[3:-2]) for k in rec if k.startswith('opp') and k.endswith('_x')))

    # Smooth opponent trajectories
    opp_xy_smooth = {}
    for oid in opp_ids:
        ox_raw = np.array(rec[f'opp{oid}_x'])
        oy_raw = np.array(rec[f'opp{oid}_y'])
        ox_s, oy_s = smooth_trajectory_xy(ox_raw, oy_raw, window=9)
        opp_xy_smooth[oid] = (ox_s, oy_s)

    # ---- Determine zoom windows ----
    if zoom_moments is not None:
        i1, i2, i3 = zoom_moments
    elif events:
        ot_idx = min(events[0]['step'], N - 1)
        i1 = max(0, ot_idx - int(N * 0.30))
        i2 = ot_idx
        i3 = min(N - 1, ot_idx + int(N * 0.25))
    else:
        i1 = int(N * 0.2)
        i2 = int(N * 0.5)
        i3 = int(N * 0.8)

    def xy_box(idx, pad=zoom_pad):
        pts_x, pts_y = [ego_x[idx]], [ego_y[idx]]
        for oid in opp_ids:
            ox_s, oy_s = opp_xy_smooth[oid]
            pts_x.append(ox_s[idx])
            pts_y.append(oy_s[idx])
        cx, cy = np.mean(pts_x), np.mean(pts_y)
        return (cx - pad, cx + pad, cy - pad, cy + pad)

    boxes = [xy_box(i1), xy_box(i2), xy_box(i3)]

    # ---- Create figure ----
    fig = plt.figure(figsize=(7.16, 2.8))
    gs = gridspec.GridSpec(1, 4, width_ratios=[2.2, 1, 1, 1],
                           wspace=0.10, left=0.04, right=0.98,
                           bottom=0.15, top=0.95)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_zooms = [fig.add_subplot(gs[0, k + 1]) for k in range(3)]

    # ---- helper: draw track (with surface fill) ----
    def draw_track(ax, lw=0.6, fill_surface=False):
        if fill_surface:
            # Fill track surface
            ax.fill(np.concatenate([xyz_l[:, 0], xyz_r[::-1, 0]]),
                    np.concatenate([xyz_l[:, 1], xyz_r[::-1, 1]]),
                    color=CLR_TRACK_FILL, alpha=0.5, zorder=0, linewidth=0)
        ax.plot(xyz_l[:, 0], xyz_l[:, 1], '-', color=CLR_TRACK_EDGE,
                linewidth=lw, zorder=2)
        ax.plot(xyz_r[:, 0], xyz_r[:, 1], '-', color=CLR_TRACK_EDGE,
                linewidth=lw, zorder=2)
        ax.plot(xyz_c[:, 0], xyz_c[:, 1], ':', color=CLR_TRACK_CL,
                linewidth=lw * 0.4, zorder=1, alpha=0.5)

    # ---- helper: draw trajectories ----
    def draw_traj(ax, labels=True, smooth_opp=True, ego_lw=1.0, opp_lw=0.7):
        lbl = 'Ego' if labels else None
        ax.plot(ego_x, ego_y, '-', color=CLR_EGO, linewidth=ego_lw,
                label=lbl, zorder=4, solid_capstyle='round')
        for oid in opp_ids:
            if smooth_opp:
                ox, oy = opp_xy_smooth[oid]
            else:
                ox = np.array(rec[f'opp{oid}_x'])
                oy = np.array(rec[f'opp{oid}_y'])
            lbl_o = 'Opponent' if (labels and oid == opp_ids[0]) else None
            ax.plot(ox, oy, '--', color=CLR_OPP, linewidth=opp_lw,
                    label=lbl_o, zorder=3, dash_capstyle='round')

    # ---- helper: time markers on overview ----
    def draw_time_markers(ax, dt=2.0, ms_e=2.5, ms_o=2.0, fs=5):
        t_marks = np.arange(0, t_arr[-1] + 0.01, dt)
        for im, tm in enumerate(t_marks):
            idx = np.argmin(np.abs(t_arr - tm))
            ax.plot(ego_x[idx], ego_y[idx], 'o', color=CLR_EGO,
                    markersize=ms_e, zorder=6,
                    markeredgecolor='white', markeredgewidth=0.3)
            ax.annotate(f'{tm:.0f}s', (ego_x[idx], ego_y[idx]),
                        fontsize=fs, ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points',
                        color=CLR_EGO, fontweight='bold')
            for oid in opp_ids:
                ox_s, oy_s = opp_xy_smooth[oid]
                ax.plot(ox_s[idx], oy_s[idx], 's', color=CLR_OPP,
                        markersize=ms_o, zorder=6,
                        markeredgecolor='white', markeredgewidth=0.2)

    # ---- helper: overtake star ----
    def draw_ot_star(ax, ms=7):
        for evt in events:
            s = evt['step']
            if s < N:
                ax.plot(ego_x[s], ego_y[s], '*', color=CLR_OT_STAR,
                        markersize=ms, zorder=8,
                        markeredgecolor='#2C3E50', markeredgewidth=0.4)

    # ============================================================
    # (a) Overview
    # ============================================================
    draw_track(ax_main, fill_surface=True)

    # Faint corridor envelope on overview
    draw_corridor_envelope_overview(ax_main, rec, track_handler,
                                    step_every=3, alpha_fill=0.05)

    draw_traj(ax_main, labels=True, ego_lw=1.0, opp_lw=0.7)
    draw_time_markers(ax_main)
    draw_ot_star(ax_main, ms=8)

    ax_main.set_aspect('equal')
    ax_main.set_xlabel('$X$ [m]')
    ax_main.set_ylabel('$Y$ [m]')
    ax_main.legend(loc='best', fontsize=6, framealpha=0.92,
                   handlelength=1.2, borderpad=0.25,
                   fancybox=True, shadow=False)

    # Auto-range with padding
    all_x = np.concatenate([xyz_l[:, 0], xyz_r[:, 0]])
    all_y = np.concatenate([xyz_l[:, 1], xyz_r[:, 1]])
    pad = 8
    ax_main.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax_main.set_ylim(all_y.min() - pad, all_y.max() + pad)

    # Draw zoom rectangles on overview (with slight transparency)
    for k, (box, zc) in enumerate(zip(boxes, ZOOM_COLORS)):
        x0, x1, y0, y1 = box
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                              linewidth=0.9, edgecolor=zc,
                              facecolor=zc, alpha=0.06,
                              linestyle='-', zorder=9)
        ax_main.add_patch(rect)
        # Number label in corner of rectangle
        ax_main.text(x0 + 1, y1 - 1, f'{chr(98+k)}',
                     fontsize=5.5, color=zc, fontweight='bold',
                     va='top', ha='left', zorder=10)

    # Sub-label below
    ax_main.text(0.5, -0.16, '(a)',
                 transform=ax_main.transAxes,
                 fontsize=8, ha='center', va='top', fontweight='bold')

    # ============================================================
    # (b)(c)(d) Zoom panels
    # ============================================================
    zoom_sub_labels = ['(b)', '(c)', '(d)']
    zoom_step_indices = [i1, i2, i3]
    zoom_desc = ['Approach', 'Overtake', 'Post-overtake']

    for k, (ax_z, box, zc, sub_lbl, step_i, desc) in enumerate(zip(
            ax_zooms, boxes, ZOOM_COLORS, zoom_sub_labels,
            zoom_step_indices, zoom_desc)):

        draw_track(ax_z, lw=0.5, fill_surface=True)

        # Draw TRUE N-step corridor at this moment
        draw_true_corridor(ax_z, rec, track_handler, step_i,
                           alpha_fill=0.20, alpha_line=0.60,
                           linewidth=0.7)

        # Draw full trajectories (thin)
        draw_traj(ax_z, labels=False, ego_lw=0.8, opp_lw=0.6)

        # Draw vehicle rectangles at this moment
        ego_hd = estimate_heading(ego_x, ego_y, step_i)
        draw_vehicle_rect(ax_z, ego_x[step_i], ego_y[step_i], ego_hd,
                          facecolor=CLR_EGO_FILL, edgecolor=CLR_EGO,
                          linewidth=0.6, draw_arrow=True, arrow_len=3.0,
                          label='Ego' if k == 0 else None)

        for oid in opp_ids:
            ox_s, oy_s = opp_xy_smooth[oid]
            opp_hd = estimate_heading(ox_s, oy_s, step_i)
            draw_vehicle_rect(ax_z, ox_s[step_i], oy_s[step_i], opp_hd,
                              facecolor=CLR_OPP_FILL, edgecolor=CLR_OPP,
                              linewidth=0.6, draw_arrow=True, arrow_len=2.5,
                              label='Opp' if k == 0 else None)

        # Decision annotation (offset below ego)
        annotate_decision(ax_z, rec, step_i,
                          ego_x[step_i], ego_y[step_i],
                          fontsize=5, offset_pts=(0, -8))

        # Time label (top-right corner)
        t_now = t_arr[step_i]
        ax_z.text(0.97, 0.97, f'$t={t_now:.1f}\\,$s',
                  transform=ax_z.transAxes, fontsize=6, ha='right', va='top',
                  bbox=dict(boxstyle='round,pad=0.12', fc='white',
                            ec='#bbbbbb', alpha=0.92, lw=0.3),
                  zorder=15)

        # Set zoom window
        x0, x1, y0, y1 = box
        ax_z.set_xlim(x0, x1)
        ax_z.set_ylim(y0, y1)
        ax_z.set_aspect('equal')
        ax_z.set_xlabel('$X$ [m]')
        if k == 0:
            ax_z.set_ylabel('$Y$ [m]')
        else:
            ax_z.set_yticklabels([])

        # Coloured border matching overview rectangle
        for spine in ax_z.spines.values():
            spine.set_edgecolor(zc)
            spine.set_linewidth(1.3)

        # Sub-label + description below
        ax_z.text(0.5, -0.16, f'{sub_lbl} {desc}',
                  transform=ax_z.transAxes,
                  fontsize=7, ha='center', va='top', fontweight='bold')

    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.03)
    plt.savefig(save_path.replace('.pdf', '.png'), format='png',
                bbox_inches='tight', pad_inches=0.03)
    plt.close()
    print(f"  Saved trajectory: {save_path}")


# ======================================================================
# 8. Time-series figure  (1×4 game-theoretic) — v3
# ======================================================================
def plot_timeseries(rec, events, save_path):
    """
    IEEE wide horizontal figure: 1×4 sub-plots.

    (a) Longitudinal gap Δs
    (b) Lateral positions + corridor
    (c) Ego decision phase (categorical)
    (d) Opponent tactic (categorical)

    v3: cleaner grid, thicker phase shading, improved legend.
    """
    setup_ieee()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    t = np.array(rec['time'])
    ego_n = np.array(rec['ego_n'])
    ego_V = np.array(rec['ego_V'])
    cl = np.array(rec['corridor_left'])
    cr = np.array(rec['corridor_right'])
    tl = np.array(rec['track_left'])
    tr_ = np.array(rec['track_right'])
    phase = np.array(rec['phase'])

    opp_ids = sorted(set(
        int(k[3:-2]) for k in rec if k.startswith('opp') and k.endswith('_s')))
    opp1 = opp_ids[0] if opp_ids else None

    opp_n_smooth = None
    if opp1 is not None:
        opp_n_smooth = smooth_n_trajectory(rec[f'opp{opp1}_n'], window=9)

    def shade_phases(ax):
        """Shade background by decision phase."""
        for pn, pc in PHASE_COLORS.items():
            m = (phase == pn)
            if not np.any(m):
                continue
            d = np.diff(m.astype(int))
            ss = np.where(d == 1)[0] + 1
            ee = np.where(d == -1)[0] + 1
            if m[0]:
                ss = np.concatenate([[0], ss])
            if m[-1]:
                ee = np.concatenate([ee, [len(m)]])
            for a, b in zip(ss, ee):
                ax.axvspan(t[a], t[min(b, len(t)-1)],
                           alpha=0.18, color=pc, linewidth=0)

    def ot_vlines(ax):
        for evt in events:
            ax.axvline(evt['time'], color=CLR_OT_STAR, linewidth=0.7,
                       linestyle='--', alpha=0.8, zorder=5)

    def style_ax(ax):
        """Common axis styling for timeseries."""
        ax.grid(True, axis='y', alpha=0.12, linewidth=0.25, linestyle=':')
        ax.grid(True, axis='x', alpha=0.08, linewidth=0.2, linestyle=':')
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.4)
        ax.spines['left'].set_linewidth(0.4)

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.0))
    fig.subplots_adjust(wspace=0.45, left=0.055, right=0.985,
                        bottom=0.24, top=0.96)

    # (a) Longitudinal gap
    ax = axes[0]
    shade_phases(ax); style_ax(ax)
    if opp1 is not None:
        gap = np.array(rec[f'opp{opp1}_gap'])
        ax.plot(t, gap, color=CLR_EGO, linewidth=0.9, solid_capstyle='round')
    ax.axhline(0, color='#2C3E50', linewidth=0.35, linestyle='-', alpha=0.35)
    ot_vlines(ax)
    ax.set_ylabel('$\\Delta s$ [m]')
    ax.set_xlabel('Time [s]')
    ax.text(0.5, -0.34, '(a) Longitudinal gap', transform=ax.transAxes,
            fontsize=7, ha='center', va='top', fontweight='bold')

    # (b) Lateral + corridor
    ax = axes[1]
    shade_phases(ax); style_ax(ax)
    ax.fill_between(t, cl, cr, alpha=0.15, color=CLR_CORRIDOR_FILL,
                    zorder=1, label='Corridor')
    ax.plot(t, tl, '-', color='#B0B0B0', linewidth=0.35, label='Track')
    ax.plot(t, tr_, '-', color='#B0B0B0', linewidth=0.35)
    ax.plot(t, cl, '-', color=CLR_CORRIDOR, linewidth=0.5, alpha=0.7)
    ax.plot(t, cr, '-', color=CLR_CORRIDOR, linewidth=0.5, alpha=0.7)
    ax.plot(t, ego_n, color=CLR_EGO, linewidth=0.9, label='Ego $n$',
            solid_capstyle='round')
    if opp_n_smooth is not None:
        ax.plot(t, opp_n_smooth, color=CLR_OPP, linewidth=0.7,
                linestyle='--', label='Opp $n$', dash_capstyle='round')
    ot_vlines(ax)
    ax.set_ylabel('Lateral $n$ [m]')
    ax.set_xlabel('Time [s]')
    ax.legend(loc='best', fontsize=4.5, ncol=2, handlelength=1.0,
              columnspacing=0.5, borderpad=0.2, labelspacing=0.25)
    ax.text(0.5, -0.34, '(b) Lateral position & corridor',
            transform=ax.transAxes,
            fontsize=7, ha='center', va='top', fontweight='bold')

    # (c) Decision phase (categorical)
    ax = axes[2]
    style_ax(ax)
    phase_list = ['RACELINE', 'SHADOW', 'OVERTAKE', 'HOLD']
    phase_y = {p: i for i, p in enumerate(phase_list)}
    phase_num = np.array([phase_y.get(p, -1) for p in phase])
    ax.step(t, phase_num, where='post', color=CLR_EGO, linewidth=0.9)
    ax.set_yticks(range(len(phase_list)))
    ax.set_yticklabels(phase_list, fontsize=5.5)
    ax.set_ylim(-0.5, len(phase_list) - 0.5)
    for pn, pc in PHASE_COLORS.items():
        if pn in phase_y:
            ax.axhspan(phase_y[pn] - 0.5, phase_y[pn] + 0.5,
                       alpha=0.20, color=pc, linewidth=0)
    ot_vlines(ax)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Ego phase')
    ax.text(0.5, -0.34, '(c) Tactical decision phase',
            transform=ax.transAxes,
            fontsize=7, ha='center', va='top', fontweight='bold')

    # (d) Opponent tactic (categorical)
    ax = axes[3]
    style_ax(ax)
    if opp1 is not None:
        opp_tac = np.array(rec[f'opp{opp1}_tactic'])
        tac_list = ['follow', 'defend_left', 'defend_right', 'yield', 'trailing']
        tac_y = {t_: i for i, t_ in enumerate(tac_list)}
        tac_num = np.array([tac_y.get(str(tc), -1) for tc in opp_tac])
        ax.step(t, tac_num, where='post', color=CLR_OPP, linewidth=0.9)
        ax.set_yticks(range(len(tac_list)))
        tac_labels_short = ['Follow', 'Def-L', 'Def-R', 'Yield', 'Trail']
        ax.set_yticklabels(tac_labels_short, fontsize=5.5)
        ax.set_ylim(-0.5, len(tac_list) - 0.5)
        tac_colors = {
            'follow': '#E8EAF6', 'defend_left': '#FFEBEE',
            'defend_right': '#FFEBEE', 'yield': '#FFF8E1',
            'trailing': '#E8F5E9',
        }
        for tn, tc in tac_colors.items():
            if tn in tac_y:
                ax.axhspan(tac_y[tn] - 0.5, tac_y[tn] + 0.5,
                           alpha=0.20, color=tc, linewidth=0)
    ot_vlines(ax)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Opponent tactic')
    ax.text(0.5, -0.34, '(d) Opponent defensive response',
            transform=ax.transAxes,
            fontsize=7, ha='center', va='top', fontweight='bold')

    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.03)
    plt.savefig(save_path.replace('.pdf', '.png'), format='png',
                bbox_inches='tight', pad_inches=0.03)
    plt.close()
    print(f"  Saved time-series: {save_path}")


# ======================================================================
# 9. GIF animation — v3 (true corridor + heading arrows)
# ======================================================================
def make_gif(rec, track_handler, save_path,
             s_min=2390, s_max=2860, fps=8):
    """GIF with true N-step corridor, vehicle rectangles, heading arrows."""
    setup_ieee()
    import matplotlib.pyplot as plt

    track_len = track_handler.s[-1]
    s_arr = track_handler.s
    mask_t = (s_arr >= s_min) & (s_arr <= s_max)
    s_reg = s_arr[mask_t]
    xyz_l = track_handler.sn2cartesian(s_reg, track_handler.w_tr_left[mask_t])
    xyz_r = track_handler.sn2cartesian(s_reg, track_handler.w_tr_right[mask_t])
    xyz_c = track_handler.sn2cartesian(s_reg, np.zeros_like(s_reg))

    ego_x = np.array(rec['ego_x'])
    ego_y = np.array(rec['ego_y'])
    t_arr = np.array(rec['time'])
    N = len(ego_x)

    opp_ids = sorted(set(
        int(k[3:-2]) for k in rec if k.startswith('opp') and k.endswith('_x')))

    opp_xy_smooth = {}
    for oid in opp_ids:
        ox_s, oy_s = smooth_trajectory_xy(
            rec[f'opp{oid}_x'], rec[f'opp{oid}_y'], window=9)
        opp_xy_smooth[oid] = (ox_s, oy_s)

    all_x = np.concatenate([xyz_l[:, 0], xyz_r[:, 0]])
    all_y = np.concatenate([xyz_l[:, 1], xyz_r[:, 1]])
    xlim = (all_x.min() - 12, all_x.max() + 12)
    ylim = (all_y.min() - 12, all_y.max() + 12)

    step_every = max(1, N // (fps * 15))
    frame_indices = list(range(0, N, step_every))

    frames = []
    for fi, idx in enumerate(frame_indices):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        # Track surface
        ax.fill(np.concatenate([xyz_l[:, 0], xyz_r[::-1, 0]]),
                np.concatenate([xyz_l[:, 1], xyz_r[::-1, 1]]),
                color=CLR_TRACK_FILL, alpha=0.5, zorder=0)
        ax.plot(xyz_l[:, 0], xyz_l[:, 1], '-', color=CLR_TRACK_EDGE, linewidth=0.8)
        ax.plot(xyz_r[:, 0], xyz_r[:, 1], '-', color=CLR_TRACK_EDGE, linewidth=0.8)
        ax.plot(xyz_c[:, 0], xyz_c[:, 1], ':', color=CLR_TRACK_CL, linewidth=0.3)

        # True corridor
        draw_true_corridor(ax, rec, track_handler, idx,
                           alpha_fill=0.15, alpha_line=0.4)

        # Trails
        trail = max(0, idx - 40)
        ax.plot(ego_x[trail:idx+1], ego_y[trail:idx+1],
                '-', color=CLR_EGO, linewidth=0.6, alpha=0.5)
        for oid in opp_ids:
            ox_s, oy_s = opp_xy_smooth[oid]
            ax.plot(ox_s[trail:idx+1], oy_s[trail:idx+1],
                    '-', color=CLR_OPP, linewidth=0.5, alpha=0.4)

        # Vehicle rectangles
        ego_hd = estimate_heading(ego_x, ego_y, idx)
        draw_vehicle_rect(ax, ego_x[idx], ego_y[idx], ego_hd,
                          facecolor=CLR_EGO_FILL, edgecolor=CLR_EGO,
                          draw_arrow=True)
        ax.annotate('Ego', (ego_x[idx], ego_y[idx]),
                    fontsize=6, ha='center', va='bottom',
                    xytext=(0, 6), textcoords='offset points',
                    color=CLR_EGO, fontweight='bold')

        for oid in opp_ids:
            ox_s, oy_s = opp_xy_smooth[oid]
            opp_hd = estimate_heading(ox_s, oy_s, idx)
            draw_vehicle_rect(ax, ox_s[idx], oy_s[idx], opp_hd,
                              facecolor=CLR_OPP_FILL, edgecolor=CLR_OPP,
                              draw_arrow=True)
            ax.annotate('Opp', (ox_s[idx], oy_s[idx]),
                        fontsize=5, ha='center', va='bottom',
                        xytext=(0, 6), textcoords='offset points',
                        color=CLR_OPP)

        ph = rec['phase'][idx] if idx < len(rec['phase']) else '?'
        cm = rec['carver_mode'][idx] if idx < len(rec['carver_mode']) else '?'
        v_ego = rec['ego_V'][idx] if idx < len(rec['ego_V']) else 0
        info_str = (f"$t = {t_arr[idx]:.1f}$ s  |  "
                    f"$V = {v_ego:.0f}$ m/s  |  {ph} · {cm}")
        ax.text(0.5, 1.02, info_str, transform=ax.transAxes,
                fontsize=8, ha='center', va='bottom', fontweight='bold')

        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('$X$ [m]')
        ax.set_ylabel('$Y$ [m]')
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frames.append(buf.reshape(h, w, 4)[:, :, :3].copy())
        plt.close(fig)
        if fi % 20 == 0:
            print(f"  GIF frame {fi+1}/{len(frame_indices)}")

    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(save_path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    print(f"  Saved GIF: {save_path}  ({len(frames)} frames)")

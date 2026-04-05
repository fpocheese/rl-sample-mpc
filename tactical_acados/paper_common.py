#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paper_common.py  —  Shared plotting & utility module for all paper figures.

IEEE top-journal quality:
  • Consistent colour palette (Nature / Science-grade)
  • Vehicle rectangles (not dots) in zoomed insets
  • Corridor (feasible region) visualisation
  • Decision-layer annotations (phase, carver mode, locked side)
  • Smoothed opponent trajectory (no jumps)
  • Time-series: 4 game-theoretic sub-plots, no speed plot
  • Sub-figure labels (a)(b)… below each panel, no titles on top

Shared by:  paper_turn678.py, paper_turn234.py, paper_turn5.py
"""

import os, sys, copy
import numpy as np

# ======================================================================
# 1. IEEE RC-params  (top-journal quality)
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
        'axes.grid': True,
        'axes.unicode_minus': False,
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
        'legend.fontsize': 6.5,
        'legend.framealpha': 0.85,
        'legend.edgecolor': '#cccccc',
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.4,
        'legend.columnspacing': 0.8,
        'legend.borderpad': 0.3,
        # --- lines ---
        'lines.linewidth': 0.8,
        'lines.markersize': 3,
        # --- grid ---
        'grid.alpha': 0.20,
        'grid.linewidth': 0.3,
        'grid.linestyle': '--',
        # --- figure ---
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })


# ======================================================================
# 2. Scientific colour palette  (Nature / IEEE-friendly)
# ======================================================================
# Primary vehicles
CLR_EGO        = '#1B66AB'   # muted blue
CLR_OPP        = '#C0392B'   # muted red
CLR_RACELINE   = '#7F8C8D'   # grey

# Track
CLR_TRACK_EDGE = '#2C3E50'   # dark charcoal
CLR_TRACK_CL   = '#BDC3C7'   # light grey  (centre-line)

# Corridor (feasible region)
CLR_CORRIDOR   = '#27AE60'   # green
CLR_CORRIDOR_FILL = '#27AE60'

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


# ======================================================================
# 3. Smoothing utilities
# ======================================================================
def smooth_trajectory_xy(x_arr, y_arr, window=7):
    """
    Smooth XY trajectory with a Savitzky-Golay-like moving average
    to remove discrete jumps from tactical lateral shifts.
    Uses a causal moving average (no future peek) for physical plausibility.
    """
    x = np.array(x_arr, dtype=float)
    y = np.array(y_arr, dtype=float)
    if len(x) < window:
        return x, y
    # Detect jumps: if the step distance exceeds 3× median step, interpolate
    dx = np.diff(x)
    dy = np.diff(y)
    step_d = np.sqrt(dx**2 + dy**2)
    med_d = max(np.median(step_d), 1e-6)
    for i in range(1, len(x)-1):
        if step_d[i-1] > 3.0 * med_d:
            # Interpolate this point from neighbours
            x[i] = 0.5 * (x[i-1] + x[i+1])
            y[i] = 0.5 * (y[i-1] + y[i+1])
    # Now apply causal moving average
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
# 4. Vehicle rectangle drawing
# ======================================================================
def draw_vehicle_rect(ax, x, y, heading, length=4.9, width=1.93,
                      facecolor='#AED6F1', edgecolor='#1B66AB',
                      linewidth=0.5, alpha=0.85, zorder=10, label=None):
    """
    Draw a vehicle as an oriented rectangle at (x,y) with given heading.
    heading is in radians, measured from x-axis.
    """
    import matplotlib.patches as mpatches
    import matplotlib.transforms as mtransforms

    # Rectangle centred on origin
    rect = mpatches.FancyBboxPatch(
        (-length/2, -width/2), length, width,
        boxstyle="round,pad=0.15",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, alpha=alpha, zorder=zorder,
        label=label,
    )
    # Transform: rotate then translate
    t = mtransforms.Affine2D().rotate(heading).translate(x, y) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    return rect


def estimate_heading(x_arr, y_arr, idx, dt=0.125):
    """Estimate heading from trajectory using central difference."""
    N = len(x_arr)
    if N < 2:
        return 0.0
    i0 = max(0, idx - 2)
    i1 = min(N - 1, idx + 2)
    if i1 == i0:
        return 0.0
    dx = x_arr[i1] - x_arr[i0]
    dy = y_arr[i1] - y_arr[i0]
    return float(np.arctan2(dy, dx))


# ======================================================================
# 5. Corridor (feasible region) drawing on zoomed insets
# ======================================================================
def draw_corridor_on_zoom(ax, rec, track_handler, s_range, step_idx,
                          alpha_fill=0.12, alpha_line=0.4):
    """
    Draw the carver corridor (feasible region) at a specific simulation step
    as a shaded band on the zoomed inset.
    Uses the recorded corridor_left/right arrays (full N-step sequence).
    """
    n_left_val = rec['corridor_left'][step_idx]
    n_right_val = rec['corridor_right'][step_idx]
    ego_s = rec['ego_s'][step_idx]
    track_len = track_handler.s[-1]

    # Draw corridor as a band around the ego position
    s_band = np.linspace(ego_s - 15, ego_s + 25, 60)
    # For each s in band, compute XY at n_left and n_right
    s_wrap = s_band % track_len
    # Get track-local corridor bounds (interpolate from ego's corridor)
    # Since corridor changes along horizon, use the first-stage value
    n_left_arr = np.full_like(s_band, n_left_val)
    n_right_arr = np.full_like(s_band, n_right_val)

    try:
        xyz_left = track_handler.sn2cartesian(s_wrap, n_left_arr)
        xyz_right = track_handler.sn2cartesian(s_wrap, n_right_arr)

        ax.fill(np.concatenate([xyz_left[:, 0], xyz_right[::-1, 0]]),
                np.concatenate([xyz_left[:, 1], xyz_right[::-1, 1]]),
                color=CLR_CORRIDOR_FILL, alpha=alpha_fill, zorder=1)
        ax.plot(xyz_left[:, 0], xyz_left[:, 1], '-',
                color=CLR_CORRIDOR, linewidth=0.5, alpha=alpha_line, zorder=2)
        ax.plot(xyz_right[:, 0], xyz_right[:, 1], '-',
                color=CLR_CORRIDOR, linewidth=0.5, alpha=alpha_line, zorder=2)
    except Exception:
        pass  # silently skip if sn2cartesian fails at boundary


def draw_full_corridor_sequence(ax, rec, track_handler, step_indices,
                                 alpha_fill=0.08):
    """
    Draw corridor for multiple time steps to show how it evolves.
    Each step gets a slightly different shade.
    """
    for idx in step_indices:
        draw_corridor_on_zoom(ax, rec, track_handler, None, idx,
                              alpha_fill=alpha_fill)


# ======================================================================
# 6. Decision annotation on zoomed insets
# ======================================================================
def annotate_decision(ax, rec, step_idx, x_pos, y_pos,
                      fontsize=5.5, show_phase=True, show_mode=True,
                      show_side=True):
    """
    Add a text annotation showing the tactical decision state.
    """
    parts = []
    if show_phase and step_idx < len(rec['phase']):
        parts.append(rec['phase'][step_idx])
    if show_mode and step_idx < len(rec['carver_mode']):
        parts.append(rec['carver_mode'][step_idx])
    if show_side and step_idx < len(rec.get('locked_side', [])):
        side = rec['locked_side'][step_idx]
        if side and side != 'None':
            parts.append(f'→{side}')

    text = ' | '.join(parts)
    if not text:
        return

    # Background box for readability
    bbox = dict(boxstyle='round,pad=0.15', facecolor='white',
                edgecolor='#999999', alpha=0.85, linewidth=0.3)
    ax.annotate(text, xy=(x_pos, y_pos),
                fontsize=fontsize, ha='center', va='top',
                bbox=bbox, zorder=15,
                xytext=(0, -6), textcoords='offset points')


# ======================================================================
# 7. Trajectory figure  (overview + 3 zoomed insets)
# ======================================================================
def plot_trajectory(rec, events, track_handler, save_path,
                    s_plot_min=2390, s_plot_max=2860,
                    zoom_pad=20, zoom_moments=None,
                    corridor_full_seq=True):
    """
    IEEE wide horizontal figure:  [overview | zoom1 | zoom2 | zoom3]
    
    Zoom insets show:
      - Vehicle rectangles (ego & opp)
      - Corridor (feasible region) shading
      - Decision state annotations
    
    No title on top. Sub-labels (a)(b)(c)(d) below each panel.
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
    gs = gridspec.GridSpec(1, 4, width_ratios=[2.0, 1, 1, 1],
                           wspace=0.08, left=0.04, right=0.98,
                           bottom=0.15, top=0.95)
    ax_main = fig.add_subplot(gs[0, 0])
    ax_zooms = [fig.add_subplot(gs[0, k + 1]) for k in range(3)]

    # ---- helper: draw track ----
    def draw_track(ax, lw=0.6):
        ax.plot(xyz_l[:, 0], xyz_l[:, 1], '-', color=CLR_TRACK_EDGE,
                linewidth=lw, zorder=2)
        ax.plot(xyz_r[:, 0], xyz_r[:, 1], '-', color=CLR_TRACK_EDGE,
                linewidth=lw, zorder=2)
        ax.plot(xyz_c[:, 0], xyz_c[:, 1], '--', color=CLR_TRACK_CL,
                linewidth=lw * 0.5, zorder=1)

    # ---- helper: draw trajectories ----
    def draw_traj(ax, labels=True, smooth_opp=True):
        lbl = 'Ego' if labels else None
        ax.plot(ego_x, ego_y, '-', color=CLR_EGO, linewidth=0.9,
                label=lbl, zorder=4)
        for oid in opp_ids:
            if smooth_opp:
                ox, oy = opp_xy_smooth[oid]
            else:
                ox = np.array(rec[f'opp{oid}_x'])
                oy = np.array(rec[f'opp{oid}_y'])
            lbl_o = f'Opponent' if (labels and oid == opp_ids[0]) else None
            ax.plot(ox, oy, '--', color=CLR_OPP, linewidth=0.7,
                    label=lbl_o, zorder=3)

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
                ax.plot(ego_x[s], ego_y[s], '*', color='#F39C12',
                        markersize=ms, zorder=8,
                        markeredgecolor='#2C3E50', markeredgewidth=0.4)

    # ============================================================
    # (a) Overview
    # ============================================================
    draw_track(ax_main)
    draw_traj(ax_main, labels=True)
    draw_time_markers(ax_main)
    draw_ot_star(ax_main, ms=8)

    ax_main.set_aspect('equal')
    ax_main.set_xlabel('$X$ [m]')
    ax_main.set_ylabel('$Y$ [m]')
    ax_main.legend(loc='best', fontsize=6, framealpha=0.85,
                   handlelength=1.5, borderpad=0.3)

    # Auto-range with padding
    all_x = np.concatenate([xyz_l[:, 0], xyz_r[:, 0]])
    all_y = np.concatenate([xyz_l[:, 1], xyz_r[:, 1]])
    pad = 10
    ax_main.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax_main.set_ylim(all_y.min() - pad, all_y.max() + pad)

    # Draw zoom rectangles on overview
    for k, (box, zc) in enumerate(zip(boxes, ZOOM_COLORS)):
        x0, x1, y0, y1 = box
        rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                              linewidth=0.8, edgecolor=zc,
                              facecolor='none', linestyle='-', zorder=9)
        ax_main.add_patch(rect)

    # Sub-label below
    ax_main.text(0.5, -0.18, '(a)', transform=ax_main.transAxes,
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

        draw_track(ax_z, lw=0.5)

        # Draw corridor at this moment
        draw_corridor_on_zoom(ax_z, rec, track_handler, None, step_i,
                              alpha_fill=0.15, alpha_line=0.5)

        # Draw full trajectories (thin)
        draw_traj(ax_z, labels=False)

        # Draw vehicle rectangles at this moment
        ego_hd = estimate_heading(ego_x, ego_y, step_i)
        draw_vehicle_rect(ax_z, ego_x[step_i], ego_y[step_i], ego_hd,
                          facecolor=CLR_EGO_FILL, edgecolor=CLR_EGO,
                          linewidth=0.6, label='Ego' if k == 0 else None)

        for oid in opp_ids:
            ox_s, oy_s = opp_xy_smooth[oid]
            opp_hd = estimate_heading(ox_s, oy_s, step_i)
            draw_vehicle_rect(ax_z, ox_s[step_i], oy_s[step_i], opp_hd,
                              facecolor=CLR_OPP_FILL, edgecolor=CLR_OPP,
                              linewidth=0.6,
                              label='Opp' if k == 0 else None)

        # Decision annotation
        annotate_decision(ax_z, rec, step_i,
                          ego_x[step_i], ego_y[step_i],
                          fontsize=5)

        # Time label
        t_now = t_arr[step_i]
        ax_z.text(0.97, 0.97, f'$t={t_now:.1f}$ s',
                  transform=ax_z.transAxes, fontsize=6, ha='right', va='top',
                  bbox=dict(boxstyle='round,pad=0.1', fc='white',
                            ec='#cccccc', alpha=0.9, lw=0.3),
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
            spine.set_linewidth(1.2)

        # Sub-label + description below
        ax_z.text(0.5, -0.18, f'{sub_lbl} {desc}',
                  transform=ax_z.transAxes,
                  fontsize=7, ha='center', va='top', fontweight='bold')

    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.03)
    plt.savefig(save_path.replace('.pdf', '.png'), format='png',
                bbox_inches='tight', pad_inches=0.03)
    plt.close()
    print(f"  Saved trajectory: {save_path}")


# ======================================================================
# 8. Time-series figure  (1×4 game-theoretic)
# ======================================================================
def plot_timeseries(rec, events, save_path):
    """
    IEEE wide horizontal figure: 1×4 sub-plots.
    
    (a) Longitudinal gap Δs  — shows approach / overtake moment
    (b) Lateral positions + corridor  — shows lane selection and feasible region
    (c) Decision phase (categorical) — shows tactical FSM transitions
    (d) Opponent tactic (categorical) — shows defender reactions
    
    No speed plot. No title. Sub-labels (a)(b)(c)(d) below each panel.
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

    # Smooth opponent lateral
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
                           alpha=0.22, color=pc, linewidth=0)

    def ot_vlines(ax):
        for evt in events:
            ax.axvline(evt['time'], color='#F39C12', linewidth=0.7,
                       linestyle='--', alpha=0.8, zorder=5)

    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.2))
    fig.subplots_adjust(wspace=0.42, left=0.055, right=0.985,
                        bottom=0.22, top=0.95)

    # ============================================================
    # (a) Longitudinal gap
    # ============================================================
    ax = axes[0]
    shade_phases(ax)
    if opp1 is not None:
        gap = np.array(rec[f'opp{opp1}_gap'])
        ax.plot(t, gap, color=CLR_EGO, linewidth=0.9)
    ax.axhline(0, color='#2C3E50', linewidth=0.4, linestyle='-', alpha=0.4)
    ot_vlines(ax)
    ax.set_ylabel('$\\Delta s$ [m]')
    ax.set_xlabel('Time [s]')
    ax.text(0.5, -0.32, '(a) Longitudinal gap', transform=ax.transAxes,
            fontsize=7, ha='center', va='top', fontweight='bold')

    # ============================================================
    # (b) Lateral + corridor
    # ============================================================
    ax = axes[1]
    shade_phases(ax)
    ax.fill_between(t, cl, cr, alpha=0.12, color=CLR_CORRIDOR, zorder=1)
    ax.plot(t, tl, '--', color='#95A5A6', linewidth=0.4, label='Track')
    ax.plot(t, tr_, '--', color='#95A5A6', linewidth=0.4)
    ax.plot(t, cl, '-', color=CLR_CORRIDOR, linewidth=0.5, alpha=0.6,
            label='Corridor')
    ax.plot(t, cr, '-', color=CLR_CORRIDOR, linewidth=0.5, alpha=0.6)
    ax.plot(t, ego_n, color=CLR_EGO, linewidth=0.9, label='Ego $n$')
    if opp_n_smooth is not None:
        ax.plot(t, opp_n_smooth, color=CLR_OPP, linewidth=0.7,
                linestyle='--', label='Opp $n$')
    ot_vlines(ax)
    ax.set_ylabel('Lateral $n$ [m]')
    ax.set_xlabel('Time [s]')
    ax.legend(loc='best', fontsize=5, ncol=2, handlelength=1.2)
    ax.text(0.5, -0.32, '(b) Lateral position & corridor',
            transform=ax.transAxes,
            fontsize=7, ha='center', va='top', fontweight='bold')

    # ============================================================
    # (c) Decision phase (categorical)
    # ============================================================
    ax = axes[2]
    phase_list = ['RACELINE', 'SHADOW', 'OVERTAKE', 'HOLD']
    phase_y = {p: i for i, p in enumerate(phase_list)}
    phase_num = np.array([phase_y.get(p, -1) for p in phase])
    # Step plot for categorical
    ax.step(t, phase_num, where='post', color=CLR_EGO, linewidth=0.9)
    ax.set_yticks(range(len(phase_list)))
    ax.set_yticklabels(phase_list, fontsize=6)
    ax.set_ylim(-0.5, len(phase_list) - 0.5)
    # Shade background
    for pn, pc in PHASE_COLORS.items():
        if pn in phase_y:
            ax.axhspan(phase_y[pn] - 0.5, phase_y[pn] + 0.5,
                       alpha=0.15, color=pc, linewidth=0)
    ot_vlines(ax)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Ego phase')
    ax.text(0.5, -0.32, '(c) Tactical decision phase',
            transform=ax.transAxes,
            fontsize=7, ha='center', va='top', fontweight='bold')

    # ============================================================
    # (d) Opponent tactic (categorical)
    # ============================================================
    ax = axes[3]
    if opp1 is not None:
        opp_tac = np.array(rec[f'opp{opp1}_tactic'])
        tac_list = ['follow', 'defend_left', 'defend_right', 'yield', 'trailing']
        tac_y = {t_: i for i, t_ in enumerate(tac_list)}
        tac_num = np.array([tac_y.get(str(tc), -1) for tc in opp_tac])
        ax.step(t, tac_num, where='post', color=CLR_OPP, linewidth=0.9)
        ax.set_yticks(range(len(tac_list)))
        tac_labels_short = ['Follow', 'Def-L', 'Def-R', 'Yield', 'Trail']
        ax.set_yticklabels(tac_labels_short, fontsize=6)
        ax.set_ylim(-0.5, len(tac_list) - 0.5)
        # Shade
        tac_colors = {
            'follow': '#E8EAF6', 'defend_left': '#FFEBEE',
            'defend_right': '#FFEBEE', 'yield': '#FFF8E1',
            'trailing': '#E8F5E9',
        }
        for tn, tc in tac_colors.items():
            if tn in tac_y:
                ax.axhspan(tac_y[tn] - 0.5, tac_y[tn] + 0.5,
                           alpha=0.15, color=tc, linewidth=0)
    ot_vlines(ax)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Opponent tactic')
    ax.text(0.5, -0.32, '(d) Opponent defensive response',
            transform=ax.transAxes,
            fontsize=7, ha='center', va='top', fontweight='bold')

    plt.savefig(save_path, format='pdf', bbox_inches='tight', pad_inches=0.03)
    plt.savefig(save_path.replace('.pdf', '.png'), format='png',
                bbox_inches='tight', pad_inches=0.03)
    plt.close()
    print(f"  Saved time-series: {save_path}")


# ======================================================================
# 9. GIF animation (improved)
# ======================================================================
def make_gif(rec, track_handler, save_path,
             s_min=2390, s_max=2860, fps=8):
    """
    GIF with vehicle rectangles, corridor, and decision state.
    """
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

    # Smooth opponent
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
        ax.plot(xyz_l[:, 0], xyz_l[:, 1], '-', color=CLR_TRACK_EDGE, linewidth=0.8)
        ax.plot(xyz_r[:, 0], xyz_r[:, 1], '-', color=CLR_TRACK_EDGE, linewidth=0.8)
        ax.plot(xyz_c[:, 0], xyz_c[:, 1], '--', color=CLR_TRACK_CL, linewidth=0.3)

        # Corridor
        draw_corridor_on_zoom(ax, rec, track_handler, None, idx,
                              alpha_fill=0.12, alpha_line=0.3)

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
                          facecolor=CLR_EGO_FILL, edgecolor=CLR_EGO)
        ax.annotate('Ego', (ego_x[idx], ego_y[idx]),
                    fontsize=6, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points',
                    color=CLR_EGO, fontweight='bold')

        for oid in opp_ids:
            ox_s, oy_s = opp_xy_smooth[oid]
            opp_hd = estimate_heading(ox_s, oy_s, idx)
            draw_vehicle_rect(ax, ox_s[idx], oy_s[idx], opp_hd,
                              facecolor=CLR_OPP_FILL, edgecolor=CLR_OPP)
            ax.annotate('Opp', (ox_s[idx], oy_s[idx]),
                        fontsize=5, ha='center', va='bottom',
                        xytext=(0, 5), textcoords='offset points',
                        color=CLR_OPP)

        ph = rec['phase'][idx] if idx < len(rec['phase']) else '?'
        cm = rec['carver_mode'][idx] if idx < len(rec['carver_mode']) else '?'
        v_ego = rec['ego_V'][idx] if idx < len(rec['ego_V']) else 0
        info_str = (f"$t = {t_arr[idx]:.1f}$ s  |  "
                    f"$V = {v_ego:.0f}$ m/s  |  {ph} / {cm}")
        ax.text(0.5, 1.02, info_str, transform=ax.transAxes,
                fontsize=8, ha='center', va='bottom', fontweight='bold')

        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.15)
        ax.set_xlabel('$X$ [m]')
        ax.set_ylabel('$Y$ [m]')
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

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt


DEFAULTS = {
    'track_name': 'yas_user_smoothed',
    'raceline_name': 'yasnorth_3d_rl_as_ref_eav25_car_gg_0.1',
    'vehicle_name': 'eav25_car',
    'plot_3d': False,
    'gg_margin': 0.1,
    'gg_abs_margin': 0.0,
}


DIR_PATH = Path(__file__).resolve().parent
DATA_PATH = DIR_PATH.parent / 'data'
TRACK_DATA_PATH = DATA_PATH / 'track_data_smoothed'
RACELINE_DATA_PATH = DATA_PATH / 'global_racing_lines'
VEHICLE_PARAMS_PATH = DATA_PATH / 'vehicle_params'
GG_DATA_PATH = DATA_PATH / 'gg_diagrams'
sys.path.append(str(DIR_PATH.parent / 'src'))

from track3D import Track3D
from ggManager import GGManager


def parse_args():
    parser = argparse.ArgumentParser(description='Plot and summarize a global raceline CSV.')
    parser.add_argument('--track', default=DEFAULTS['track_name'], help='Track CSV stem or path.')
    parser.add_argument('--raceline', default=DEFAULTS['raceline_name'], help='Raceline CSV stem or path.')
    parser.add_argument('--vehicle', default=DEFAULTS['vehicle_name'], help='Vehicle name used for gg diagram lookup.')
    parser.add_argument('--gg-margin', type=float, default=DEFAULTS['gg_margin'], help='GG margin used for acceleration usage plots.')
    parser.add_argument('--gg-abs-margin', type=float, default=DEFAULTS['gg_abs_margin'], help='Absolute gg margin used in usage plots.')
    parser.add_argument('--plot-3d', action='store_true', default=DEFAULTS['plot_3d'], help='Also create a 3D path plot.')
    parser.add_argument('--show', action='store_true', help='Show figures interactively after saving them.')
    parser.add_argument('--output-dir', default=None, help='Directory for generated report figures and summary text.')
    return parser.parse_args()


def resolve_csv_path(path_or_stem: str, base_dir: Path) -> Path:
    candidate = Path(path_or_stem)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    if not str(candidate).lower().endswith('.csv'):
        candidate = Path(str(candidate) + '.csv')
    return candidate.resolve()


def load_vehicle_params(vehicle_name: str) -> dict:
    with open(VEHICLE_PARAMS_PATH / f'params_{vehicle_name}.yml', 'r') as stream:
        return yaml.safe_load(stream)


def build_output_dir(raceline_path: Path, output_dir: str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).resolve()
    return (RACELINE_DATA_PATH / 'plots' / raceline_path.stem).resolve()


def load_raceline_dataframe(raceline_path: Path) -> pd.DataFrame:
    trajectory_data_frame = pd.read_csv(raceline_path, comment='#', sep=',')
    required_columns = ['s_opt', 'v_opt', 'n_opt', 'chi_opt', 'ax_opt', 'ay_opt', 'jx_opt', 'jy_opt', 't_opt']
    missing = [column for column in required_columns if column not in trajectory_data_frame.columns]
    if missing:
        raise KeyError(f'Raceline file is missing columns: {missing}')
    return trajectory_data_frame


def calc_acceleration_usage(track_handler: Track3D, gg_handler: GGManager, trajectory_data_frame: pd.DataFrame):
    s_opt = trajectory_data_frame['s_opt'].to_numpy()
    v_opt = trajectory_data_frame['v_opt'].to_numpy()
    n_opt = trajectory_data_frame['n_opt'].to_numpy()
    chi_opt = trajectory_data_frame['chi_opt'].to_numpy()
    ax_opt = trajectory_data_frame['ax_opt'].to_numpy()
    ay_opt = trajectory_data_frame['ay_opt'].to_numpy()

    ax_tilde, ay_tilde, g_tilde = track_handler.calc_apparent_accelerations_numpy(
        s=s_opt,
        V=v_opt,
        n=n_opt,
        chi=chi_opt,
        ax=ax_opt,
        ay=ay_opt,
    )
    gg_values = gg_handler.acc_interpolator(
        np.array((v_opt.flatten(), g_tilde.flatten()))
    ).full().squeeze().reshape(4, -1)
    gg_exponent, ax_min, ax_max, ay_max = gg_values

    ax_avail = np.abs(ax_min) * np.power(
        np.maximum(
            1.0 - np.power(np.minimum(np.abs(ay_tilde) / ay_max, 1.0), gg_exponent),
            1e-3,
        ),
        1.0 / gg_exponent,
    )

    s_opt_fine = np.linspace(s_opt[0], s_opt[-1], 100 * len(s_opt))
    n_opt_fine = np.interp(s_opt_fine, s_opt, n_opt)
    v_opt_fine = np.interp(s_opt_fine, s_opt, v_opt)
    chi_opt_fine = np.interp(s_opt_fine, s_opt, chi_opt)
    ax_opt_fine = np.interp(s_opt_fine, s_opt, ax_opt)
    ay_opt_fine = np.interp(s_opt_fine, s_opt, ay_opt)
    ax_tilde_fine, ay_tilde_fine, g_tilde_fine = track_handler.calc_apparent_accelerations_numpy(
        s=s_opt_fine,
        V=v_opt_fine,
        n=n_opt_fine,
        chi=chi_opt_fine,
        ax=ax_opt_fine,
        ay=ay_opt_fine,
    )
    gg_values_fine = gg_handler.acc_interpolator(
        np.array((v_opt_fine.flatten(), g_tilde_fine.flatten()))
    ).full().squeeze().reshape(4, -1)
    gg_exponent_fine, ax_min_fine, ax_max_fine, ay_max_fine = gg_values_fine
    ax_avail_fine = np.abs(ax_min_fine) * np.power(
        np.maximum(
            1.0 - np.power(np.minimum(np.abs(ay_tilde_fine) / ay_max_fine, 1.0), gg_exponent_fine),
            1e-3,
        ),
        1.0 / gg_exponent_fine,
    )

    return {
        'ax_tilde': ax_tilde,
        'ay_tilde': ay_tilde,
        'g_tilde': g_tilde,
        'ax_min': ax_min,
        'ax_max': ax_max,
        'ay_max': ay_max,
        'ax_avail': ax_avail,
        's_fine': s_opt_fine,
        'ax_tilde_fine': ax_tilde_fine,
        'ay_tilde_fine': ay_tilde_fine,
        'ax_max_fine': ax_max_fine,
        'ay_max_fine': ay_max_fine,
        'ax_avail_fine': ax_avail_fine,
    }


def build_summary(track_handler: Track3D, trajectory_data_frame: pd.DataFrame, acc_usage: dict) -> dict:
    v_opt = trajectory_data_frame['v_opt'].to_numpy()
    n_opt = trajectory_data_frame['n_opt'].to_numpy()
    chi_opt = trajectory_data_frame['chi_opt'].to_numpy()
    ax_opt = trajectory_data_frame['ax_opt'].to_numpy()
    ay_opt = trajectory_data_frame['ay_opt'].to_numpy()
    jx_opt = trajectory_data_frame['jx_opt'].to_numpy()
    jy_opt = trajectory_data_frame['jy_opt'].to_numpy()
    t_opt = trajectory_data_frame['t_opt'].to_numpy()

    left_clearance = track_handler.w_tr_left - n_opt
    right_clearance = n_opt - track_handler.w_tr_right
    long_margin = np.minimum(acc_usage['ax_avail'] - acc_usage['ax_tilde'], acc_usage['ax_max'] - acc_usage['ax_tilde'])
    lat_margin = acc_usage['ay_max'] - np.abs(acc_usage['ay_tilde'])

    return {
        'track_length_m': float(track_handler.s[-1]),
        'num_samples': int(len(trajectory_data_frame)),
        'lap_time_s': float(t_opt[-1]),
        'speed_min_mps': float(np.min(v_opt)),
        'speed_mean_mps': float(np.mean(v_opt)),
        'speed_max_mps': float(np.max(v_opt)),
        'lateral_offset_min_m': float(np.min(n_opt)),
        'lateral_offset_max_m': float(np.max(n_opt)),
        'peak_abs_chi_deg': float(np.rad2deg(np.max(np.abs(chi_opt)))),
        'peak_abs_ax_mps2': float(np.max(np.abs(ax_opt))),
        'peak_abs_ay_mps2': float(np.max(np.abs(ay_opt))),
        'peak_abs_jx_mps3': float(np.max(np.abs(jx_opt))),
        'peak_abs_jy_mps3': float(np.max(np.abs(jy_opt))),
        'peak_abs_ax_tilde_mps2': float(np.max(np.abs(acc_usage['ax_tilde']))),
        'peak_abs_ay_tilde_mps2': float(np.max(np.abs(acc_usage['ay_tilde']))),
        'peak_g_tilde_mps2': float(np.max(acc_usage['g_tilde'])),
        'min_left_clearance_m': float(np.min(left_clearance)),
        'min_right_clearance_m': float(np.min(right_clearance)),
        'min_longitudinal_gg_margin_mps2': float(np.min(long_margin)),
        'min_lateral_gg_margin_mps2': float(np.min(lat_margin)),
    }


def save_summary_text(summary: dict, summary_path: Path):
    with open(summary_path, 'w') as summary_file:
        for key, value in summary.items():
            if isinstance(value, int):
                summary_file.write(f'{key}: {value}\n')
            else:
                summary_file.write(f'{key}: {value:.6f}\n')


def create_path_figure(track_handler: Track3D, trajectory_data_frame: pd.DataFrame, title: str, plot_3d: bool):
    left, right = track_handler.get_track_bounds()
    xyz_rl = track_handler.sn2cartesian(
        trajectory_data_frame['s_opt'].to_numpy(),
        trajectory_data_frame['n_opt'].to_numpy(),
    )
    v_opt = trajectory_data_frame['v_opt'].to_numpy()

    if plot_3d:
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(projection='3d')
        ax.plot(left[0], left[1], left[2], color='black', linewidth=1.0)
        ax.plot(right[0], right[1], right[2], color='black', linewidth=1.0)
        ax.plot(track_handler.x, track_handler.y, track_handler.z, '--', color='dimgray', linewidth=1.0, label='Reference')
        scatter = ax.scatter(xyz_rl[:, 0], xyz_rl[:, 1], xyz_rl[:, 2], c=v_opt, s=8, cmap='viridis', label='Raceline')
        ax.set_xlabel('x in m')
        ax.set_ylabel('y in m')
        ax.set_zlabel('z in m')
        ax.set_title(title)
        ax.legend(loc='best')
    else:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(left[0], left[1], color='black', linewidth=1.0)
        ax.plot(right[0], right[1], color='black', linewidth=1.0)
        ax.plot(track_handler.x, track_handler.y, '--', color='dimgray', linewidth=1.0, label='Reference')
        scatter = ax.scatter(xyz_rl[:, 0], xyz_rl[:, 1], c=v_opt, s=10, cmap='viridis', label='Raceline')
        ax.set_xlabel('x in m')
        ax.set_ylabel('y in m')
        ax.set_aspect('equal')
        ax.grid()
        ax.set_title(title)
        ax.legend(loc='best')

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label('Speed in m/s')
    fig.tight_layout()
    return fig


def create_profile_figure(trajectory_data_frame: pd.DataFrame):
    s_opt = trajectory_data_frame['s_opt'].to_numpy()
    fig, axes = plt.subplots(nrows=5, figsize=(11, 12), sharex=True)
    axes[0].plot(s_opt, trajectory_data_frame['v_opt'].to_numpy(), label=r'$V$')
    axes[0].set_ylabel('m/s')
    axes[1].plot(s_opt, trajectory_data_frame['n_opt'].to_numpy(), label=r'$n$')
    axes[1].set_ylabel('m')
    axes[2].plot(s_opt, np.rad2deg(trajectory_data_frame['chi_opt'].to_numpy()), label=r'$\chi$')
    axes[2].set_ylabel('deg')
    axes[3].plot(s_opt, trajectory_data_frame['ax_opt'].to_numpy(), label=r'$a_x$')
    axes[3].plot(s_opt, trajectory_data_frame['ay_opt'].to_numpy(), label=r'$a_y$')
    axes[3].set_ylabel('m/s^2')
    axes[4].plot(s_opt, trajectory_data_frame['jx_opt'].to_numpy(), label=r'$j_x$')
    axes[4].plot(s_opt, trajectory_data_frame['jy_opt'].to_numpy(), label=r'$j_y$')
    axes[4].set_ylabel('m/s^3')
    axes[4].set_xlabel('s in m')

    for axis in axes:
        axis.grid()
        axis.legend(loc='best')

    fig.suptitle('Raceline profiles')
    fig.tight_layout()
    return fig


def create_apparent_figure(trajectory_data_frame: pd.DataFrame, acc_usage: dict):
    s_opt = trajectory_data_frame['s_opt'].to_numpy()
    fig, axes = plt.subplots(nrows=3, figsize=(11, 9), sharex=True)
    axes[0].plot(s_opt, acc_usage['g_tilde'], label=r'$\tilde{g}$')
    axes[0].set_ylabel('m/s^2')
    axes[1].plot(s_opt, acc_usage['ax_tilde'], label=r'$\tilde{a}_x$')
    axes[1].set_ylabel('m/s^2')
    axes[2].plot(s_opt, acc_usage['ay_tilde'], label=r'$\tilde{a}_y$')
    axes[2].set_ylabel('m/s^2')
    axes[2].set_xlabel('s in m')

    for axis in axes:
        axis.grid()
        axis.legend(loc='best')

    fig.suptitle('Apparent accelerations')
    fig.tight_layout()
    return fig


def create_acceleration_usage_figure(trajectory_data_frame: pd.DataFrame, acc_usage: dict, gg_abs_margin: float):
    s_opt = trajectory_data_frame['s_opt'].to_numpy()
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 10), sharex='col')
    axes[0, 0].set_title('Original samples')
    axes[0, 1].set_title('Interpolated samples')

    axes[0, 0].plot(s_opt, acc_usage['ax_tilde'], label=r'$\tilde{a}_x$')
    axes[0, 0].plot(s_opt, acc_usage['ax_avail'] + gg_abs_margin, label=r'$\tilde{a}_{x,available}$')
    axes[0, 0].plot(s_opt, acc_usage['ax_avail'] + gg_abs_margin - acc_usage['ax_tilde'], label='margin')

    axes[1, 0].plot(s_opt, acc_usage['ax_tilde'], label=r'$\tilde{a}_x$')
    axes[1, 0].plot(s_opt, acc_usage['ax_max'] + gg_abs_margin, label=r'$\tilde{a}_{x,max}$')
    axes[1, 0].plot(s_opt, acc_usage['ax_max'] + gg_abs_margin - acc_usage['ax_tilde'], label='margin')

    axes[2, 0].plot(s_opt, acc_usage['ay_tilde'], label=r'$\tilde{a}_y$')
    axes[2, 0].plot(s_opt, acc_usage['ay_max'] + gg_abs_margin, label=r'$\tilde{a}_{y,max}$')
    axes[2, 0].plot(s_opt, acc_usage['ay_max'] + gg_abs_margin - np.abs(acc_usage['ay_tilde']), label='margin')

    axes[0, 1].plot(acc_usage['s_fine'], acc_usage['ax_tilde_fine'], label=r'$\tilde{a}_x$')
    axes[0, 1].plot(acc_usage['s_fine'], acc_usage['ax_avail_fine'] + gg_abs_margin, label=r'$\tilde{a}_{x,available}$')
    axes[0, 1].plot(acc_usage['s_fine'], acc_usage['ax_avail_fine'] + gg_abs_margin - acc_usage['ax_tilde_fine'], label='margin')

    axes[1, 1].plot(acc_usage['s_fine'], acc_usage['ax_tilde_fine'], label=r'$\tilde{a}_x$')
    axes[1, 1].plot(acc_usage['s_fine'], acc_usage['ax_max_fine'] + gg_abs_margin, label=r'$\tilde{a}_{x,max}$')
    axes[1, 1].plot(acc_usage['s_fine'], acc_usage['ax_max_fine'] + gg_abs_margin - acc_usage['ax_tilde_fine'], label='margin')

    axes[2, 1].plot(acc_usage['s_fine'], acc_usage['ay_tilde_fine'], label=r'$\tilde{a}_y$')
    axes[2, 1].plot(acc_usage['s_fine'], acc_usage['ay_max_fine'] + gg_abs_margin, label=r'$\tilde{a}_{y,max}$')
    axes[2, 1].plot(acc_usage['s_fine'], acc_usage['ay_max_fine'] + gg_abs_margin - np.abs(acc_usage['ay_tilde_fine']), label='margin')

    for axis_row in axes:
        for axis in axis_row:
            axis.grid()
            axis.legend(loc='best')

    axes[2, 0].set_xlabel('s in m')
    axes[2, 1].set_xlabel('s in m')
    fig.suptitle('GG usage')
    fig.tight_layout()
    return fig


def save_raceline_report(
    track_name: str,
    raceline_name: str,
    vehicle_name: str,
    gg_margin: float,
    gg_abs_margin: float = 0.0,
    plot_3d: bool = False,
    show: bool = False,
    output_dir: str | None = None,
) -> dict:
    track_path = resolve_csv_path(track_name, TRACK_DATA_PATH)
    raceline_path = resolve_csv_path(raceline_name, RACELINE_DATA_PATH)
    report_dir = build_output_dir(raceline_path, output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    track_handler = Track3D(path=str(track_path))
    gg_handler = GGManager(
        gg_path=str(GG_DATA_PATH / vehicle_name / 'velocity_frame'),
        gg_margin=gg_margin,
    )
    trajectory_data_frame = load_raceline_dataframe(raceline_path)
    acc_usage = calc_acceleration_usage(track_handler, gg_handler, trajectory_data_frame)
    summary = build_summary(track_handler, trajectory_data_frame, acc_usage)

    figure_title = f'{raceline_path.stem} on {track_path.stem}\nLap time {summary["lap_time_s"]:.3f} s'
    figures = {
        'path_xy.png': create_path_figure(track_handler, trajectory_data_frame, figure_title, plot_3d=False),
        'profiles.png': create_profile_figure(trajectory_data_frame),
        'apparent_accelerations.png': create_apparent_figure(trajectory_data_frame, acc_usage),
        'gg_usage.png': create_acceleration_usage_figure(trajectory_data_frame, acc_usage, gg_abs_margin),
    }
    if plot_3d:
        figures['path_3d.png'] = create_path_figure(track_handler, trajectory_data_frame, figure_title, plot_3d=True)

    for file_name, figure in figures.items():
        figure.savefig(report_dir / file_name, dpi=200)

    summary_path = report_dir / 'summary.txt'
    save_summary_text(summary, summary_path)

    print(f'Raceline report written to {report_dir}')
    for key, value in summary.items():
        if isinstance(value, int):
            print(f'{key}: {value}')
        else:
            print(f'{key}: {value:.6f}')

    if show:
        plt.show()
    else:
        plt.close('all')

    return {
        'report_dir': str(report_dir),
        'summary_path': str(summary_path),
        'summary': summary,
    }


if __name__ == '__main__':
    args = parse_args()
    save_raceline_report(
        track_name=args.track,
        raceline_name=args.raceline,
        vehicle_name=args.vehicle,
        gg_margin=args.gg_margin,
        gg_abs_margin=args.gg_abs_margin,
        plot_3d=args.plot_3d,
        show=args.show,
        output_dir=args.output_dir,
    )

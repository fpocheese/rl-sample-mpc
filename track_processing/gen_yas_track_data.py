import os
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.spatial import cKDTree


# settings
reference_baseline = 'BaseLineSim.csv'
shift_cm_dir_name = 'shift_cm'
left_boundary_file_cm = 'yasnorth_tbs_left_enushifted.csv'
right_boundary_file_cm = 'yasnorth_tbs_right_enushifted.csv'
left_boundary_file_m = 'yasnorth_tbs_left_enushifted_m.csv'
right_boundary_file_m = 'yasnorth_tbs_right_enushifted_m.csv'

track_bounds_output_file_name = 'yasnorth_bounds_3d'
track_data_output_file_name = 'yasnorth_3d'
comparison_plot_file_name = 'yasnorth_boundary_comparison.png'

reference_step_size = 2.0  # in meter
boundary_resample_step_size = 1.0  # in meter
ray_margin = 5.0  # in meter
run_smoothing = True
smoothing_method = 'boundary_aware_spline'  # boundary_aware_spline, track3d_nlp
spline_xy_smoothing_factor = 5.0
spline_z_smoothing_factor = 0.25
spline_dense_step_size = 0.5
spline_midpoint_iterations = 2
show_plots = False

# Dictionary for cost function of track smoothing.
weights = {
    'w_c': 1e0,  # deviation to measurements centerline
    'w_l': 1e0,  # deviation to measurements left bound
    'w_r': 1e0,  # deviation to measurements right bound
    'w_theta': 1e7,  # smoothness theta
    'w_mu': 1e5,  # smoothness mu
    'w_phi': 1e4,  # smoothness phi
    'w_nl': 1e-2,  # smoothness left bound
    'w_nr': 1e-2  # smoothness right bound
}

# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
yas_data_path = os.path.join(data_path, 'yas_data')
shift_cm_path = os.path.join(yas_data_path, shift_cm_dir_name)
track_bounds_path = os.path.join(data_path, 'track_bounds')
track_data_path = os.path.join(data_path, 'track_data')
track_data_smoothed_path = os.path.join(data_path, 'track_data_smoothed')
comparison_plot_path = os.path.join(yas_data_path, comparison_plot_file_name)
os.makedirs(track_bounds_path, exist_ok=True)
os.makedirs(track_data_path, exist_ok=True)
os.makedirs(track_data_smoothed_path, exist_ok=True)
sys.path.append(os.path.join(dir_path, '..', 'src'))

from track3D import Track3D


def close_polyline(points: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    if np.linalg.norm(points[0] - points[-1]) > tol:
        return np.vstack((points, points[0]))
    points_closed = points.copy()
    points_closed[-1] = points_closed[0]
    return points_closed


def load_boundary_csv(path: str, scale: float = 1.0) -> np.ndarray:
    return pd.read_csv(path)[['X', 'Y', 'Z']].to_numpy() * scale


def save_boundary_csv(points: np.ndarray, out_path: str):
    pd.DataFrame(points, columns=['X', 'Y', 'Z']).to_csv(
        out_path,
        sep=',',
        index=False,
        float_format='%.6f'
    )


def cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(segment_lengths)))


def resample_polyline(points: np.ndarray, step_size: float) -> np.ndarray:
    s_in = cumulative_arc_length(points)
    n_samples = max(int(np.ceil(s_in[-1] / step_size)), 2)
    s_out = np.linspace(0.0, s_in[-1], n_samples + 1)
    out = np.column_stack([
        np.interp(s_out, s_in, points[:, dim]) for dim in range(points.shape[1])
    ])
    out[-1] = out[0]
    return out


def wrap_to_reference(angle: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return reference + (angle - reference + np.pi) % (2.0 * np.pi) - np.pi


def cross2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def intersect_ray_with_polyline(origin_xy: np.ndarray, direction_xy: np.ndarray, polyline_xyz: np.ndarray, ray_length: float):
    ray = direction_xy * ray_length
    seg_start = polyline_xyz[:-1, :2]
    seg_end = polyline_xyz[1:, :2]
    seg = seg_end - seg_start
    q_minus_p = seg_start - origin_xy
    denom = cross2d(ray, seg)

    valid = np.abs(denom) > 1e-9
    if not np.any(valid):
        return None

    t = np.full(seg.shape[0], np.inf)
    u = np.full(seg.shape[0], np.inf)
    t[valid] = cross2d(q_minus_p[valid], seg[valid]) / denom[valid]
    u[valid] = cross2d(q_minus_p[valid], ray) / denom[valid]
    valid &= (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)

    if not np.any(valid):
        return None

    idx_candidates = np.where(valid)[0]
    idx = idx_candidates[np.argmin(t[idx_candidates])]
    intersection_xy = origin_xy + t[idx] * ray
    intersection_z = polyline_xyz[idx, 2] + u[idx] * (polyline_xyz[idx + 1, 2] - polyline_xyz[idx, 2])
    distance = np.linalg.norm(intersection_xy - origin_xy)

    return np.array([intersection_xy[0], intersection_xy[1], intersection_z]), distance


def closest_point_on_polyline(point_xy: np.ndarray, polyline_xyz: np.ndarray):
    seg_start = polyline_xyz[:-1]
    seg_end = polyline_xyz[1:]
    seg_vec = seg_end[:, :2] - seg_start[:, :2]
    seg_len_sq = np.sum(seg_vec ** 2, axis=1)
    seg_len_sq = np.where(seg_len_sq < 1e-12, 1e-12, seg_len_sq)
    rel = point_xy - seg_start[:, :2]
    alpha = np.sum(rel * seg_vec, axis=1) / seg_len_sq
    alpha = np.clip(alpha, 0.0, 1.0)
    candidates_xy = seg_start[:, :2] + alpha[:, np.newaxis] * seg_vec
    distances_sq = np.sum((candidates_xy - point_xy) ** 2, axis=1)
    idx = int(np.argmin(distances_sq))
    point_xyz = seg_start[idx] + alpha[idx] * (seg_end[idx] - seg_start[idx])
    return point_xyz


def align_heading(theta_ref: np.ndarray, x_ref: np.ndarray, y_ref: np.ndarray):
    tangent = np.diff(np.column_stack((x_ref, y_ref)), axis=0)
    tangent = np.vstack((tangent, tangent[-1]))
    theta_num = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    theta_ref = wrap_to_reference(theta_ref, theta_num)
    return theta_ref


def build_reference_from_baseline(path: str, step_size: float):
    baseline = pd.read_csv(path)
    xy_closed = close_polyline(baseline[['Xref', 'Yref']].to_numpy())
    s_closed = cumulative_arc_length(xy_closed)

    theta_raw = np.unwrap(baseline['Aref'].to_numpy())
    theta_closed = np.concatenate((theta_raw, [theta_raw[0] + 2.0 * np.pi]))
    kappa_closed = np.concatenate((baseline['Kref'].to_numpy(), [baseline['Kref'].iloc[0]]))
    width_left_closed = np.concatenate((baseline['Lmax'].to_numpy(), [baseline['Lmax'].iloc[0]]))
    width_right_closed = np.concatenate((baseline['Lmin'].to_numpy(), [baseline['Lmin'].iloc[0]]))

    n_samples = max(int(np.ceil(s_closed[-1] / step_size)), 2)
    s_ref = np.linspace(0.0, s_closed[-1], n_samples + 1)
    x_ref = np.interp(s_ref, s_closed, xy_closed[:, 0])
    y_ref = np.interp(s_ref, s_closed, xy_closed[:, 1])
    theta_ref = np.interp(s_ref, s_closed, theta_closed)
    theta_ref = align_heading(theta_ref=theta_ref, x_ref=x_ref, y_ref=y_ref)
    kappa_ref = np.interp(s_ref, s_closed, kappa_closed)
    width_left_ref = np.interp(s_ref, s_closed, width_left_closed)
    width_right_ref = np.interp(s_ref, s_closed, width_right_closed)

    return {
        's': s_ref,
        'x': x_ref,
        'y': y_ref,
        'theta': theta_ref,
        'kappa': kappa_ref,
        'width_left': width_left_ref,
        'width_right': width_right_ref
    }


def intersect_bounds(reference: dict, left_boundary_xyz: np.ndarray, right_boundary_xyz: np.ndarray, ray_length: float):
    n_points = reference['s'].size
    left_points = np.zeros((n_points, 3))
    right_points = np.zeros((n_points, 3))
    z_ref = np.zeros(n_points)
    left_fallback = np.zeros(n_points, dtype=bool)
    right_fallback = np.zeros(n_points, dtype=bool)

    for i in range(n_points):
        origin_xy = np.array([reference['x'][i], reference['y'][i]])
        normal_xy = np.array([
            -np.sin(reference['theta'][i]),
            np.cos(reference['theta'][i])
        ])
        normal_xy /= np.linalg.norm(normal_xy)

        left_hit = intersect_ray_with_polyline(
            origin_xy=origin_xy,
            direction_xy=normal_xy,
            polyline_xyz=left_boundary_xyz,
            ray_length=ray_length
        )
        right_hit = intersect_ray_with_polyline(
            origin_xy=origin_xy,
            direction_xy=-normal_xy,
            polyline_xyz=right_boundary_xyz,
            ray_length=ray_length
        )

        if left_hit is None:
            left_fallback[i] = True
            left_guess_xy = origin_xy + normal_xy * reference['width_left'][i]
            left_points[i] = closest_point_on_polyline(left_guess_xy, left_boundary_xyz)
        else:
            left_points[i] = left_hit[0]

        if right_hit is None:
            right_fallback[i] = True
            right_guess_xy = origin_xy + normal_xy * reference['width_right'][i]
            right_points[i] = closest_point_on_polyline(right_guess_xy, right_boundary_xyz)
        else:
            right_points[i] = right_hit[0]

        z_ref[i] = 0.5 * (left_points[i, 2] + right_points[i, 2])

    left_points[-1] = left_points[0]
    right_points[-1] = right_points[0]
    z_ref[-1] = z_ref[0]

    return left_points, right_points, z_ref, left_fallback, right_fallback


def to_open_points(points: np.ndarray) -> np.ndarray:
    return close_polyline(points)[:-1].copy()


def to_closed_points(points: np.ndarray) -> np.ndarray:
    return close_polyline(points)


def resample_open_polyline(points_open: np.ndarray, step_size: float) -> np.ndarray:
    return resample_polyline(to_closed_points(points_open), step_size)[:-1]


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def periodic_first_derivative(values: np.ndarray, step_size: float) -> np.ndarray:
    return (np.roll(values, -1) - np.roll(values, 1)) / (2.0 * step_size)


def periodic_angle_derivative(values: np.ndarray, step_size: float) -> np.ndarray:
    return wrap_angle(np.roll(values, -1) - np.roll(values, 1)) / (2.0 * step_size)


def periodic_tangent(points_open: np.ndarray) -> np.ndarray:
    diff = np.roll(points_open, -1, axis=0) - np.roll(points_open, 1, axis=0)
    return diff / np.linalg.norm(diff, axis=1, keepdims=True)


def fit_periodic_spline(s_samples: np.ndarray, values: np.ndarray, smoothing_factor: float):
    return interpolate.splrep(s_samples, values, s=smoothing_factor, per=1, k=3)


def smooth_centerline_with_splines(track_raw: Track3D, step_size: float) -> np.ndarray:
    reference_xyz = np.column_stack((track_raw.x, track_raw.y, track_raw.z))
    reference_xyz = to_open_points(reference_xyz)
    s_samples = track_raw.s[:-1]

    spline_x = fit_periodic_spline(s_samples, reference_xyz[:, 0], spline_xy_smoothing_factor)
    spline_y = fit_periodic_spline(s_samples, reference_xyz[:, 1], spline_xy_smoothing_factor)
    spline_z = fit_periodic_spline(s_samples, reference_xyz[:, 2], spline_z_smoothing_factor)

    s_dense = np.linspace(0.0, s_samples[-1], int(np.ceil(s_samples[-1] / spline_dense_step_size)) + 1)
    centerline_dense = np.column_stack((
        interpolate.splev(s_dense, spline_x),
        interpolate.splev(s_dense, spline_y),
        interpolate.splev(s_dense, spline_z),
    ))
    centerline_dense[-1] = centerline_dense[0]

    return resample_open_polyline(centerline_dense[:-1], step_size)


def intersect_reference_with_bounds(
        reference_open_xyz: np.ndarray,
        normal_open_xy: np.ndarray,
        left_boundary_xyz: np.ndarray,
        right_boundary_xyz: np.ndarray,
        ray_length: float,
        fallback_left: np.ndarray | None = None,
        fallback_right: np.ndarray | None = None,
):
    n_points = reference_open_xyz.shape[0]
    left_points = np.zeros((n_points, 3))
    right_points = np.zeros((n_points, 3))
    left_fallback = np.zeros(n_points, dtype=bool)
    right_fallback = np.zeros(n_points, dtype=bool)

    fallback_left = fallback_left if fallback_left is not None else np.full(n_points, ray_length * 0.5)
    fallback_right = fallback_right if fallback_right is not None else np.full(n_points, ray_length * 0.5)

    for i in range(n_points):
        origin_xy = reference_open_xyz[i, :2]
        normal_xy = normal_open_xy[i] / np.linalg.norm(normal_open_xy[i])

        left_hit = intersect_ray_with_polyline(
            origin_xy=origin_xy,
            direction_xy=normal_xy,
            polyline_xyz=left_boundary_xyz,
            ray_length=ray_length
        )
        right_hit = intersect_ray_with_polyline(
            origin_xy=origin_xy,
            direction_xy=-normal_xy,
            polyline_xyz=right_boundary_xyz,
            ray_length=ray_length
        )

        if left_hit is None:
            left_fallback[i] = True
            left_guess_xy = origin_xy + normal_xy * fallback_left[i]
            left_points[i] = closest_point_on_polyline(left_guess_xy, left_boundary_xyz)
        else:
            left_points[i] = left_hit[0]

        if right_hit is None:
            right_fallback[i] = True
            right_guess_xy = origin_xy - normal_xy * fallback_right[i]
            right_points[i] = closest_point_on_polyline(right_guess_xy, right_boundary_xyz)
        else:
            right_points[i] = right_hit[0]

    return left_points, right_points, left_fallback, right_fallback


def smooth_track_boundary_aware(
        track_raw_path: str,
        left_boundary_xyz: np.ndarray,
        right_boundary_xyz: np.ndarray,
        out_path: str,
        step_size: float,
        ray_length: float,
):
    track_raw = Track3D(path=track_raw_path)
    centerline_open = smooth_centerline_with_splines(track_raw=track_raw, step_size=step_size)

    # Re-center the smoothed reference between the measured boundaries to avoid
    # high-curvature corner cutting.
    for _ in range(spline_midpoint_iterations):
        tangent_open = periodic_tangent(centerline_open)
        normal_open_xy = np.column_stack((-tangent_open[:, 1], tangent_open[:, 0]))
        left_points, right_points, _, _ = intersect_reference_with_bounds(
            reference_open_xyz=centerline_open,
            normal_open_xy=normal_open_xy,
            left_boundary_xyz=left_boundary_xyz,
            right_boundary_xyz=right_boundary_xyz,
            ray_length=ray_length,
        )
        midpoint_open = 0.5 * (left_points + right_points)
        centerline_open = resample_open_polyline(midpoint_open, step_size)

    tangent_open = periodic_tangent(centerline_open)
    normal_open_xy = np.column_stack((-tangent_open[:, 1], tangent_open[:, 0]))
    left_points, right_points, left_fallback, right_fallback = intersect_reference_with_bounds(
        reference_open_xyz=centerline_open,
        normal_open_xy=normal_open_xy,
        left_boundary_xyz=left_boundary_xyz,
        right_boundary_xyz=right_boundary_xyz,
        ray_length=ray_length,
    )

    reference_open_xyz = resample_open_polyline(0.5 * (left_points + right_points), step_size)
    tangent_open = periodic_tangent(reference_open_xyz)
    theta_open = np.unwrap(np.arctan2(tangent_open[:, 1], tangent_open[:, 0]))
    mu_open = -np.arcsin(np.clip(tangent_open[:, 2], -1.0, 1.0))

    normal_open_xy = np.column_stack((-tangent_open[:, 1], tangent_open[:, 0]))
    left_points, right_points, left_fallback_final, right_fallback_final = intersect_reference_with_bounds(
        reference_open_xyz=reference_open_xyz,
        normal_open_xy=normal_open_xy,
        left_boundary_xyz=left_boundary_xyz,
        right_boundary_xyz=right_boundary_xyz,
        ray_length=ray_length,
    )

    lateral_vector = left_points - right_points
    lateral_vector -= np.sum(lateral_vector * tangent_open, axis=1, keepdims=True) * tangent_open
    normal_open = lateral_vector / np.linalg.norm(lateral_vector, axis=1, keepdims=True)

    flat_normal = np.column_stack((-np.sin(theta_open), np.cos(theta_open), np.zeros_like(theta_open)))
    bank_basis = np.column_stack((
        np.cos(theta_open) * np.sin(mu_open),
        np.sin(theta_open) * np.sin(mu_open),
        np.cos(mu_open)
    ))
    phi_open = np.arctan2(
        np.sum(normal_open * bank_basis, axis=1),
        np.sum(normal_open * flat_normal, axis=1)
    )

    reference_open_xyz = 0.5 * (left_points + right_points)
    step_size_actual = np.mean(np.diff(cumulative_arc_length(to_closed_points(reference_open_xyz))))
    dtheta_open = periodic_angle_derivative(theta_open, step_size_actual)
    dmu_open = periodic_first_derivative(mu_open, step_size_actual)
    dphi_open = periodic_first_derivative(phi_open, step_size_actual)
    w_tr_left_open = np.sum((left_points - reference_open_xyz) * normal_open, axis=1)
    w_tr_right_open = np.sum((right_points - reference_open_xyz) * normal_open, axis=1)

    omega_open = np.zeros((reference_open_xyz.shape[0], 3))
    for i in range(reference_open_xyz.shape[0]):
        jacobian = Track3D.get_jacobian_J(mu_open[i], phi_open[i])
        omega_open[i] = jacobian.dot(np.array([dphi_open[i], dmu_open[i], dtheta_open[i]]))

    reference_closed_xyz = to_closed_points(reference_open_xyz)
    s_closed = cumulative_arc_length(reference_closed_xyz)
    theta_lap_count = int(np.round((theta_open[-1] - theta_open[0]) / (2.0 * np.pi)))

    track_data_frame = pd.DataFrame()
    track_data_frame['s_m'] = s_closed
    track_data_frame['x_m'] = reference_closed_xyz[:, 0]
    track_data_frame['y_m'] = reference_closed_xyz[:, 1]
    track_data_frame['z_m'] = reference_closed_xyz[:, 2]
    track_data_frame['theta_rad'] = np.concatenate((theta_open, [theta_open[0] + 2.0 * np.pi * theta_lap_count]))
    track_data_frame['mu_rad'] = np.concatenate((mu_open, [mu_open[0]]))
    track_data_frame['phi_rad'] = np.concatenate((phi_open, [phi_open[0]]))
    track_data_frame['dtheta_radpm'] = np.concatenate((dtheta_open, [dtheta_open[0]]))
    track_data_frame['dmu_radpm'] = np.concatenate((dmu_open, [dmu_open[0]]))
    track_data_frame['dphi_radpm'] = np.concatenate((dphi_open, [dphi_open[0]]))
    track_data_frame['w_tr_right_m'] = np.concatenate((w_tr_right_open, [w_tr_right_open[0]]))
    track_data_frame['w_tr_left_m'] = np.concatenate((w_tr_left_open, [w_tr_left_open[0]]))
    track_data_frame['omega_x_radpm'] = np.concatenate((omega_open[:, 0], [omega_open[0, 0]]))
    track_data_frame['omega_y_radpm'] = np.concatenate((omega_open[:, 1], [omega_open[0, 1]]))
    track_data_frame['omega_z_radpm'] = np.concatenate((omega_open[:, 2], [omega_open[0, 2]]))
    track_data_frame.to_csv(out_path, sep=',', index=False, float_format='%.6f')

    fallback_stats = {
        'left': int(np.sum(left_fallback) + np.sum(left_fallback_final)),
        'right': int(np.sum(right_fallback) + np.sum(right_fallback_final)),
    }

    return Track3D(path=out_path), fallback_stats


def save_track_bounds(left_points: np.ndarray, right_points: np.ndarray, out_path: str):
    track_bound_frame = pd.DataFrame()
    track_bound_frame['right_bound_x'] = right_points[:, 0]
    track_bound_frame['right_bound_y'] = right_points[:, 1]
    track_bound_frame['right_bound_z'] = right_points[:, 2]
    track_bound_frame['left_bound_x'] = left_points[:, 0]
    track_bound_frame['left_bound_y'] = left_points[:, 1]
    track_bound_frame['left_bound_z'] = left_points[:, 2]
    track_bound_frame.to_csv(out_path, sep=',', index=False, float_format='%.6f')


def print_boundary_error_stats(label: str, reconstructed_bounds: np.ndarray, original_bounds_xyz: np.ndarray):
    tree = cKDTree(original_bounds_xyz[:, :2])
    distances, _ = tree.query(reconstructed_bounds[:, :2])
    print(
        f'{label}: mean 2D error = {np.mean(distances):.3f} m, '
        f'max 2D error = {np.max(distances):.3f} m'
    )


def plot_comparison(
        left_original_xyz: np.ndarray,
        right_original_xyz: np.ndarray,
        left_original_dense_xyz: np.ndarray,
        right_original_dense_xyz: np.ndarray,
        track_raw: Track3D,
        track_smoothed,
        reference_xyz: np.ndarray,
        left_fallback: np.ndarray,
        right_fallback: np.ndarray,
        out_path: str,
        show: bool,
):
    left_raw, right_raw = track_raw.get_track_bounds()

    print_boundary_error_stats('Raw left reconstruction', left_raw.T, left_original_dense_xyz)
    print_boundary_error_stats('Raw right reconstruction', right_raw.T, right_original_dense_xyz)
    if track_smoothed is not None:
        left_smoothed, right_smoothed = track_smoothed.get_track_bounds()
        print_boundary_error_stats('Smoothed left reconstruction', left_smoothed.T, left_original_dense_xyz)
        print_boundary_error_stats('Smoothed right reconstruction', right_smoothed.T, right_original_dense_xyz)
    else:
        left_smoothed = None
        right_smoothed = None

    fig = plt.figure(figsize=(14, 6))
    ax_xy = fig.add_subplot(121)
    ax_3d = fig.add_subplot(122, projection='3d')

    # original bounds
    ax_xy.plot(left_original_xyz[:, 0], left_original_xyz[:, 1], color='black', linewidth=1.0, label='Original Left')
    ax_xy.plot(right_original_xyz[:, 0], right_original_xyz[:, 1], color='dimgray', linewidth=1.0, label='Original Right')
    ax_3d.plot(left_original_xyz[:, 0], left_original_xyz[:, 1], left_original_xyz[:, 2], color='black', linewidth=1.0)
    ax_3d.plot(right_original_xyz[:, 0], right_original_xyz[:, 1], right_original_xyz[:, 2], color='dimgray', linewidth=1.0)

    # reconstructed raw bounds
    ax_xy.plot(left_raw[0], left_raw[1], '--', color='tab:orange', linewidth=1.0, label='Raw Reconstructed Left')
    ax_xy.plot(right_raw[0], right_raw[1], '--', color='tab:blue', linewidth=1.0, label='Raw Reconstructed Right')
    ax_3d.plot(left_raw[0], left_raw[1], left_raw[2], '--', color='tab:orange', linewidth=1.0)
    ax_3d.plot(right_raw[0], right_raw[1], right_raw[2], '--', color='tab:blue', linewidth=1.0)

    if left_smoothed is not None and right_smoothed is not None:
        ax_xy.plot(left_smoothed[0], left_smoothed[1], color='tab:red', linewidth=1.2, label='Smoothed Reconstructed Left')
        ax_xy.plot(right_smoothed[0], right_smoothed[1], color='tab:green', linewidth=1.2, label='Smoothed Reconstructed Right')
        ax_3d.plot(left_smoothed[0], left_smoothed[1], left_smoothed[2], color='tab:red', linewidth=1.2)
        ax_3d.plot(right_smoothed[0], right_smoothed[1], right_smoothed[2], color='tab:green', linewidth=1.2)

    # reference line
    ax_xy.plot(reference_xyz[:, 0], reference_xyz[:, 1], color='tab:purple', linewidth=1.0, label='Reference Line')
    ax_3d.plot(reference_xyz[:, 0], reference_xyz[:, 1], reference_xyz[:, 2], color='tab:purple', linewidth=1.0)

    if np.any(left_fallback):
        ax_xy.scatter(reference_xyz[left_fallback, 0], reference_xyz[left_fallback, 1], color='tab:red', s=10, label='Left Fallback')
    if np.any(right_fallback):
        ax_xy.scatter(reference_xyz[right_fallback, 0], reference_xyz[right_fallback, 1], color='tab:green', s=10, label='Right Fallback')

    ax_xy.set_title('YAS Boundary Conversion Check')
    ax_xy.set_xlabel('x')
    ax_xy.set_ylabel('y')
    ax_xy.axis('equal')
    ax_xy.grid()
    ax_xy.legend(loc='upper right')

    ax_3d.set_title('3D View')
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('z')
    ax_3d.set_box_aspect((
        np.ptp(reference_xyz[:, 0]),
        np.ptp(reference_xyz[:, 1]),
        max(np.ptp(reference_xyz[:, 2]), 1.0)
    ))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


baseline_path = os.path.join(yas_data_path, reference_baseline)
left_boundary_path_cm = os.path.join(shift_cm_path, left_boundary_file_cm)
right_boundary_path_cm = os.path.join(shift_cm_path, right_boundary_file_cm)
left_boundary_path_m = os.path.join(yas_data_path, left_boundary_file_m)
right_boundary_path_m = os.path.join(yas_data_path, right_boundary_file_m)

track_bounds_output_path = os.path.join(track_bounds_path, track_bounds_output_file_name + '.csv')
track_data_output_path = os.path.join(track_data_path, track_data_output_file_name + '.csv')
track_data_smoothed_output_path = os.path.join(track_data_smoothed_path, track_data_output_file_name + '_smoothed.csv')

reference = build_reference_from_baseline(
    path=baseline_path,
    step_size=reference_step_size
)

left_boundary_original = close_polyline(load_boundary_csv(left_boundary_path_cm, scale=0.01))
right_boundary_original = close_polyline(load_boundary_csv(right_boundary_path_cm, scale=0.01))

save_boundary_csv(left_boundary_original[:-1], left_boundary_path_m)
save_boundary_csv(right_boundary_original[:-1], right_boundary_path_m)

left_boundary_dense = resample_polyline(left_boundary_original, boundary_resample_step_size)
right_boundary_dense = resample_polyline(right_boundary_original, boundary_resample_step_size)

ray_length = max(
    np.max(reference['width_left']),
    np.max(np.abs(reference['width_right']))
) + ray_margin

left_points, right_points, z_ref, left_fallback, right_fallback = intersect_bounds(
    reference=reference,
    left_boundary_xyz=left_boundary_dense,
    right_boundary_xyz=right_boundary_dense,
    ray_length=ray_length
)

reference_xyz = np.column_stack((reference['x'], reference['y'], z_ref))
reference_xyz[-1] = reference_xyz[0]

save_track_bounds(left_points=left_points, right_points=right_points, out_path=track_bounds_output_path)

print(f'Left fallback count: {np.sum(left_fallback)} / {left_fallback.size}')
print(f'Right fallback count: {np.sum(right_fallback)} / {right_fallback.size}')

track_handler = Track3D()
track_handler.generate_3d_from_3d_track_bounds(
    path=track_bounds_output_path,
    out_path=track_data_output_path,
    reference=reference_xyz,
    ignore_banking=False,
    visualize=False
)

track_raw = Track3D(path=track_data_output_path)
track_smoothed = None
smoothed_fallback_stats = None

if run_smoothing:
    if smoothing_method == 'boundary_aware_spline':
        track_smoothed, smoothed_fallback_stats = smooth_track_boundary_aware(
            track_raw_path=track_data_output_path,
            left_boundary_xyz=left_boundary_dense,
            right_boundary_xyz=right_boundary_dense,
            out_path=track_data_smoothed_output_path,
            step_size=reference_step_size,
            ray_length=ray_length,
        )
    elif smoothing_method == 'track3d_nlp':
        track_handler = Track3D()
        track_handler.smooth_track(
            out_path=track_data_smoothed_output_path,
            weights=weights,
            step_size=reference_step_size,
            visualize=False,
            in_path=track_data_output_path
        )
        track_smoothed = Track3D(path=track_data_smoothed_output_path)
    else:
        raise ValueError(f'Unsupported smoothing_method: {smoothing_method}')

plot_comparison(
    left_original_xyz=left_boundary_original,
    right_original_xyz=right_boundary_original,
    left_original_dense_xyz=left_boundary_dense,
    right_original_dense_xyz=right_boundary_dense,
    track_raw=track_raw,
    track_smoothed=track_smoothed,
    reference_xyz=reference_xyz,
    left_fallback=left_fallback,
    right_fallback=right_fallback,
    out_path=comparison_plot_path,
    show=show_plots
)

print(f'Track bounds saved to {track_bounds_output_path}')
print(f'Raw track data saved to {track_data_output_path}')
if run_smoothing:
    print(f'Smoothed track data saved to {track_data_smoothed_output_path}')
    if smoothed_fallback_stats is not None:
        print(
            f"Smoothed track fallback count: left={smoothed_fallback_stats['left']}, "
            f"right={smoothed_fallback_stats['right']}"
        )
else:
    print('Smoothed track data skipped. Set run_smoothing = True to enable it.')
print(f'Converted left boundary saved to {left_boundary_path_m}')
print(f'Converted right boundary saved to {right_boundary_path_m}')
print(f'Comparison plot saved to {comparison_plot_path}')

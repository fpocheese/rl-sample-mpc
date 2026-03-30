import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


DIR_PATH = Path(__file__).resolve().parent
REPO_ROOT = DIR_PATH.parent
sys.path.append(str(REPO_ROOT / "src"))

from track3D import Track3D


def load_boundary(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    required = {"X", "Y", "Z"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    return frame[["X", "Y", "Z"]].to_numpy()


def reconstruct_track_bounds(track_path: Path) -> dict:
    handler = Track3D(path=str(track_path))
    center = np.column_stack((handler.x, handler.y, handler.z))
    left = handler.sn2cartesian(handler.s, handler.w_tr_left)
    right = handler.sn2cartesian(handler.s, handler.w_tr_right)
    return {
        "handler": handler,
        "center": center,
        "left": np.asarray(left),
        "right": np.asarray(right),
    }


def nearest_distance_stats(reference_xyz: np.ndarray, query_xyz: np.ndarray) -> dict:
    tree = cKDTree(reference_xyz[:, :2])
    distances, _ = tree.query(query_xyz[:, :2], k=1)
    return {
        "distances": distances,
        "mean": float(np.mean(distances)),
        "max": float(np.max(distances)),
        "p95": float(np.percentile(distances, 95.0)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare smoothed track files against original boundaries.")
    parser.add_argument("--left-boundary", required=True, type=Path)
    parser.add_argument("--right-boundary", required=True, type=Path)
    parser.add_argument("--track-a", required=True, type=Path)
    parser.add_argument("--track-b", required=True, type=Path)
    parser.add_argument("--label-a", default="track_a")
    parser.add_argument("--label-b", default="track_b")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--zoom-padding", type=float, default=60.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    left_boundary = load_boundary(args.left_boundary)
    right_boundary = load_boundary(args.right_boundary)

    track_a = reconstruct_track_bounds(args.track_a)
    track_b = reconstruct_track_bounds(args.track_b)

    stats = {
        args.label_a: {
            "left": nearest_distance_stats(left_boundary, track_a["left"]),
            "right": nearest_distance_stats(right_boundary, track_a["right"]),
            "length_m": float(track_a["handler"].s[-1]),
            "samples": int(track_a["handler"].s.shape[0]),
            "max_abs_omega_z": float(np.max(np.abs(track_a["handler"].Omega_z))),
        },
        args.label_b: {
            "left": nearest_distance_stats(left_boundary, track_b["left"]),
            "right": nearest_distance_stats(right_boundary, track_b["right"]),
            "length_m": float(track_b["handler"].s[-1]),
            "samples": int(track_b["handler"].s.shape[0]),
            "max_abs_omega_z": float(np.max(np.abs(track_b["handler"].Omega_z))),
        },
    }

    # Use the worse point between the two reconstructions to generate a zoom window.
    worst_label = args.label_a
    worst_side = "left"
    worst_dist = stats[worst_label][worst_side]["max"]
    for label in (args.label_a, args.label_b):
        for side in ("left", "right"):
            if stats[label][side]["max"] > worst_dist:
                worst_label = label
                worst_side = side
                worst_dist = stats[label][side]["max"]

    worst_track = track_a if worst_label == args.label_a else track_b
    boundary_xyz = left_boundary if worst_side == "left" else right_boundary
    reconstructed_xyz = worst_track[worst_side]
    dists = nearest_distance_stats(boundary_xyz, reconstructed_xyz)["distances"]
    worst_idx = int(np.argmax(dists))
    zoom_center = reconstructed_xyz[worst_idx, :2]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_xy = axes[0, 0]
    ax_zoom = axes[0, 1]
    ax_left = axes[1, 0]
    ax_right = axes[1, 1]

    ax_xy.plot(left_boundary[:, 0], left_boundary[:, 1], color="black", lw=1.8, label="original left boundary")
    ax_xy.plot(right_boundary[:, 0], right_boundary[:, 1], color="dimgray", lw=1.8, label="original right boundary")
    ax_xy.plot(track_a["left"][:, 0], track_a["left"][:, 1], color="tab:blue", alpha=0.9, label=f"{args.label_a} left")
    ax_xy.plot(track_a["right"][:, 0], track_a["right"][:, 1], color="tab:blue", alpha=0.9, ls="--", label=f"{args.label_a} right")
    ax_xy.plot(track_a["center"][:, 0], track_a["center"][:, 1], color="tab:blue", lw=1.0, alpha=0.8, label=f"{args.label_a} center")
    ax_xy.plot(track_b["left"][:, 0], track_b["left"][:, 1], color="tab:orange", alpha=0.9, label=f"{args.label_b} left")
    ax_xy.plot(track_b["right"][:, 0], track_b["right"][:, 1], color="tab:orange", alpha=0.9, ls="--", label=f"{args.label_b} right")
    ax_xy.plot(track_b["center"][:, 0], track_b["center"][:, 1], color="tab:orange", lw=1.0, alpha=0.8, label=f"{args.label_b} center")
    ax_xy.set_title("Full XY Overlay")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(loc="best", fontsize=9)

    ax_zoom.plot(left_boundary[:, 0], left_boundary[:, 1], color="black", lw=1.8)
    ax_zoom.plot(right_boundary[:, 0], right_boundary[:, 1], color="dimgray", lw=1.8)
    ax_zoom.plot(track_a["left"][:, 0], track_a["left"][:, 1], color="tab:blue", alpha=0.9)
    ax_zoom.plot(track_a["right"][:, 0], track_a["right"][:, 1], color="tab:blue", alpha=0.9, ls="--")
    ax_zoom.plot(track_b["left"][:, 0], track_b["left"][:, 1], color="tab:orange", alpha=0.9)
    ax_zoom.plot(track_b["right"][:, 0], track_b["right"][:, 1], color="tab:orange", alpha=0.9, ls="--")
    ax_zoom.scatter(zoom_center[0], zoom_center[1], color="red", s=28, label=f"worst area: {worst_label} {worst_side}")
    ax_zoom.set_xlim(zoom_center[0] - args.zoom_padding, zoom_center[0] + args.zoom_padding)
    ax_zoom.set_ylim(zoom_center[1] - args.zoom_padding, zoom_center[1] + args.zoom_padding)
    ax_zoom.set_title("Zoom Near Worst Boundary Mismatch")
    ax_zoom.set_aspect("equal", adjustable="box")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.legend(loc="best", fontsize=9)

    ax_left.plot(track_a["handler"].s, stats[args.label_a]["left"]["distances"], color="tab:blue", label=args.label_a)
    ax_left.plot(track_b["handler"].s, stats[args.label_b]["left"]["distances"], color="tab:orange", label=args.label_b)
    ax_left.set_title("Left Boundary XY Error")
    ax_left.set_xlabel("s [m]")
    ax_left.set_ylabel("nearest error [m]")
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(loc="best")

    ax_right.plot(track_a["handler"].s, stats[args.label_a]["right"]["distances"], color="tab:blue", label=args.label_a)
    ax_right.plot(track_b["handler"].s, stats[args.label_b]["right"]["distances"], color="tab:orange", label=args.label_b)
    ax_right.set_title("Right Boundary XY Error")
    ax_right.set_xlabel("s [m]")
    ax_right.set_ylabel("nearest error [m]")
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(loc="best")

    summary_lines = []
    for label in (args.label_a, args.label_b):
        summary_lines.append(
            f"{label}: samples={stats[label]['samples']}, length={stats[label]['length_m']:.1f} m, "
            f"max|omega_z|={stats[label]['max_abs_omega_z']:.4f} rad/m, "
            f"left mean/max={stats[label]['left']['mean']:.3f}/{stats[label]['left']['max']:.3f} m, "
            f"right mean/max={stats[label]['right']['mean']:.3f}/{stats[label]['right']['max']:.3f} m"
        )
    fig.suptitle("\n".join(summary_lines), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(f"Saved figure to {args.output}")
    for label in (args.label_a, args.label_b):
        print(summary_lines[0] if label == args.label_a else summary_lines[1])


if __name__ == "__main__":
    main()

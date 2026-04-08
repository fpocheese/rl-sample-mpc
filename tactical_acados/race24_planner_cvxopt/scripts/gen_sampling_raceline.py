#!/usr/bin/env python3
"""
Generate a sampling-planner-ready CSV from the existing RaceLine CSV.

Input CSV columns (your OCP output):
  Sref, Xref, Yref, Aref, Kref, Lmax, Lmin,
  S, L, X, Y, A, dA, K, V, AN, AT, Vs, ANs, ATs, TIME

Output CSV adds 4 new columns:
  s_dot   = V * cos(chi) / (1 - n * kappa_ref)
  n_dot   = V * sin(chi)
  s_ddot  = d(s_dot)/dt  (from ax, ay, chi, n, kappa_ref, dkappa_ref/ds)
  n_ddot  = d(n_dot)/dt  (from ax, chi)

Usage:
  python3 gen_sampling_raceline.py <input.csv> [output.csv]

If output path is omitted, writes to <input>_sampling.csv
"""
import sys
import numpy as np
import pandas as pd


def compute_frenet_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """Add s_dot, n_dot, s_ddot, n_ddot columns."""

    # Use 'optimised' velocity/accel columns
    V   = df['Vs'].values.astype(float)
    chi = df['dA'].values.astype(float)     # chi = A - Aref
    n   = df['L'].values.astype(float)      # lateral offset
    s   = df['S'].values.astype(float)      # frenet arc-length
    ax  = df['ATs'].values.astype(float)    # longitudinal accel (velocity frame)
    ay  = df['ANs'].values.astype(float)    # lateral accel (velocity frame)
    kref = df['Kref'].values.astype(float)  # centreline curvature
    t   = df['TIME'].values.astype(float)   # time

    N = len(V)

    # ---- s_dot = V cos(chi) / (1 - n * kref) ----
    denom = 1.0 - n * kref
    denom[np.abs(denom) < 1e-6] = 1e-6
    s_dot = V * np.cos(chi) / denom

    # ---- n_dot = V sin(chi) ----
    n_dot = V * np.sin(chi)

    # ---- Compute dkref/ds numerically (periodic) ----
    # s is periodic with period s[-1] (approximately)
    ds = np.diff(s)
    ds[ds < 1e-6] = 1e-6
    dkref_ds = np.zeros(N)
    for i in range(1, N - 1):
        dkref_ds[i] = (kref[i + 1] - kref[i - 1]) / (s[i + 1] - s[i - 1]) if (s[i + 1] - s[i - 1]) > 1e-6 else 0.0
    dkref_ds[0] = dkref_ds[1]
    dkref_ds[-1] = dkref_ds[-2]

    # ---- s_ddot and n_ddot from analytical expressions ----
    # chi_dot = ay / V - Omega_z * s_dot
    V_safe = V.copy()
    V_safe[np.abs(V_safe) < 1e-3] = 1e-3
    chi_dot = ay / V_safe - kref * s_dot

    # s_ddot = [ (ax*cos(chi) - V*sin(chi)*chi_dot) * (1-n*kref)
    #            - V*cos(chi) * (-n_dot*kref - n*dkref_ds*s_dot) ] / (1-n*kref)^2
    one_minus_nk = 1.0 - n * kref
    one_minus_nk_sq = one_minus_nk ** 2
    one_minus_nk_sq[np.abs(one_minus_nk_sq) < 1e-6] = 1e-6

    s_ddot = (
        (ax * np.cos(chi) - V * np.sin(chi) * chi_dot) * one_minus_nk
        - V * np.cos(chi) * (-n_dot * kref - n * dkref_ds * s_dot)
    ) / one_minus_nk_sq

    # n_ddot = ax * sin(chi) + V * cos(chi) * chi_dot
    n_ddot = ax * np.sin(chi) + V * np.cos(chi) * chi_dot

    df = df.copy()
    df['s_dot']  = s_dot
    df['n_dot']  = n_dot
    df['s_ddot'] = s_ddot
    df['n_ddot'] = n_ddot

    return df


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    in_path = sys.argv[1]
    if len(sys.argv) >= 3:
        out_path = sys.argv[2]
    else:
        out_path = in_path.replace('.csv', '_sampling.csv')

    print(f"Reading: {in_path}")
    df = pd.read_csv(in_path)

    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    df = compute_frenet_derivatives(df)

    # Sanity checks
    print(f"\n  s_dot  range: [{df['s_dot'].min():.4f}, {df['s_dot'].max():.4f}]")
    print(f"  n_dot  range: [{df['n_dot'].min():.6f}, {df['n_dot'].max():.6f}]")
    print(f"  s_ddot range: [{df['s_ddot'].min():.4f}, {df['s_ddot'].max():.4f}]")
    print(f"  n_ddot range: [{df['n_ddot'].min():.6f}, {df['n_ddot'].max():.6f}]")

    # Compare s_dot vs Vs (should be close for small chi and n)
    rel_diff = np.abs(df['s_dot'].values - df['Vs'].values) / np.maximum(df['Vs'].values, 1e-3)
    print(f"  |s_dot - Vs|/Vs: mean={rel_diff.mean():.6f}, max={rel_diff.max():.6f}")

    df.to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}")
    print(f"  New columns: s_dot, n_dot, s_ddot, n_ddot")


if __name__ == '__main__':
    main()

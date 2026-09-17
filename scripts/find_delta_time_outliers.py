#!/usr/bin/env python3
"""Scan experiment_1/24022026 for metrics.csv files containing a delta_time
row far from the GLOBAL distribution.

Strategy:
  1. First pass: collect all delta_time values across every metrics.csv to build
     the global mean/std (and median/MAD as a robust fallback).
  2. Second pass: list files that have at least one row with |z| >= threshold
     (default z=5.0, using robust median/MAD-based z-score).

Usage:
    python find_delta_time_outliers.py
    python find_delta_time_outliers.py --z 6
    python find_delta_time_outliers.py --root experiment_1/24022026
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path


def read_delta_times(csv_path):
    values = []
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "delta_time" not in reader.fieldnames:
                return []
            for row in reader:
                raw = row.get("delta_time", "")
                if not raw or raw.lower() == "nan":
                    continue
                try:
                    values.append(float(raw))
                except ValueError:
                    continue
    except OSError as e:
        print(f"[warn] could not read {csv_path}: {e}", file=sys.stderr)
    return values


def scan(root, z_thr):
    root = Path(root)
    if not root.exists():
        print(f"[error] root does not exist: {root}", file=sys.stderr)
        return 2

    files = sorted(root.rglob("metrics.csv"))
    if not files:
        print(f"[info] no metrics.csv found under {root}")
        return 0

    per_file = []
    all_values = []
    for fp in files:
        vs = read_delta_times(fp)
        if vs:
            per_file.append((fp, vs))
            all_values.extend(vs)

    if not all_values:
        print("[info] no delta_time values found")
        return 0

    median = statistics.median(all_values)
    mad = statistics.median(abs(v - median) for v in all_values)
    scale = 1.4826 * mad if mad > 0 else statistics.pstdev(all_values)
    if scale <= 0:
        print("[info] delta_time has zero spread; nothing to flag")
        return 0

    print(f"[global] files={len(per_file)} rows={len(all_values)} "
          f"median={median:.4f} mad={mad:.4f} scale={scale:.4f} "
          f"min={min(all_values):.4f} max={max(all_values):.4f}")
    print(f"[threshold] flagging rows with |v - median| / scale >= {z_thr}")
    print("-" * 80)

    flagged = 0
    for fp, vs in per_file:
        worst_v = max(vs, key=lambda v: abs(v - median))
        worst_z = abs(worst_v - median) / scale
        if worst_z < z_thr:
            continue
        flagged += 1
        rel = fp.relative_to(root) if fp.is_absolute() else fp
        print(f"{rel}\tworst={worst_v:.3f}\tz={worst_z:.1f}")

    print("-" * 80)
    print(f"[summary] {flagged}/{len(per_file)} files have a delta_time row with z>={z_thr}")
    return 0


def main():
    default_root = Path(__file__).resolve().parent.parent / "experiment_1" / "24022026"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(default_root),
                    help="directory to scan recursively for metrics.csv")
    ap.add_argument("--z", type=float, default=5.0,
                    help="robust z-score threshold (median/MAD), default 5.0")
    args = ap.parse_args()
    return scan(args.root, args.z)


if __name__ == "__main__":
    raise SystemExit(main())

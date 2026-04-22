"""
find_short_runs.py
------------------
Scans the ablation directory tree and reports any metrics.csv files
whose episode count is shorter than the expected maximum.

Usage:
    python find_short_runs.py [base_dir] [--expected N] [--target metrics.csv]

Defaults:
    base_dir   = ./test/333
    --expected = auto (uses the max found across all files)
    --target   = metrics.csv
"""

import os
import csv
import argparse
from collections import defaultdict
from typing import List, Tuple, Dict


# ── helpers ──────────────────────────────────────────────────────────────────

def count_episodes(path: str) -> int:
    """Return the number of data rows in a CSV (= number of episodes logged)."""
    try:
        with open(path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)          # skip header
            return sum(1 for _ in reader)
    except Exception as e:
        return -1                       # mark as unreadable


def find_all_metrics(base: str, target_tail: str) -> List[str]:
    """Walk *base* and collect every file whose path ends with *target_tail*."""
    found = []
    for dirpath, _, filenames in os.walk(base):
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            if full.replace("\\", "/").endswith(target_tail):
                found.append(full)
    return found


def classify_by_filter(path: str, filter_names: List[str]) -> str:
    """Return the first filter name found in the path, or 'UNKNOWN'."""
    for name in filter_names:
        if name in path:
            return name
    return "UNKNOWN"


# ── main logic ────────────────────────────────────────────────────────────────

def find_short_runs(
    base_dir: str,
    target_tail: str = "metrics.csv",
    expected: int = None,
    filter_names: List[str] = None,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Returns a dict  {filter_key: [(path, episode_count), ...]}
    for every run that is shorter than *expected* episodes.
    """
    all_paths = find_all_metrics(base_dir, target_tail)

    if not all_paths:
        print(f"[WARN] No files matching '*/{target_tail}' found under '{base_dir}'")
        return {}

    # Count episodes for every file
    counts: List[Tuple[str, int]] = []
    for p in all_paths:
        n = count_episodes(p)
        counts.append((p, n))

    valid_counts = [n for _, n in counts if n >= 0]
    if not valid_counts:
        print("[ERROR] Could not read any file.")
        return {}

    max_eps  = max(valid_counts)
    min_eps  = min(valid_counts)
    mean_eps = sum(valid_counts) / len(valid_counts)

    threshold = expected if expected is not None else max_eps

    print("=" * 70)
    print(f"  Base dir : {base_dir}")
    print(f"  Target   : {target_tail}")
    print(f"  Files    : {len(counts)}")
    print(f"  Episodes → max={max_eps}  min={min_eps}  mean={mean_eps:.1f}")
    print(f"  Threshold: {threshold}  (runs below this are flagged)")
    print("=" * 70)

    # Group short runs
    short: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    for path, n in sorted(counts, key=lambda x: x[1]):
        if n < threshold:
            key = classify_by_filter(path, filter_names or [])
            short[key].append((path, n))

    if not short:
        print("\n✅  All runs have the full expected number of episodes.\n")
        return {}

    # ── pretty report ────────────────────────────────────────────────────────
    total_short = sum(len(v) for v in short.values())
    print(f"\n⚠️   {total_short} short run(s) found:\n")

    for group, items in sorted(short.items()):
        print(f"  [{group}]  ({len(items)} file(s))")
        for path, n in items:
            missing = threshold - n
            rel     = os.path.relpath(path, base_dir)
            bar_ok  = "█" * int(30 * n / threshold)
            bar_no  = "░" * (30 - len(bar_ok))
            pct     = 100 * n / threshold
            print(f"    {pct:5.1f}%  [{bar_ok}{bar_no}]  ep={n:4d}  missing={missing:4d}")
            print(f"           {rel}")
        print()

    # ── episode distribution ─────────────────────────────────────────────────
    print("-" * 70)
    print("  Episode count distribution (all files):")
    buckets: Dict[int, int] = defaultdict(int)
    bucket_size = max(1, max_eps // 10)
    for _, n in counts:
        if n >= 0:
            buckets[(n // bucket_size) * bucket_size] += 1

    for lo in sorted(buckets):
        hi  = lo + bucket_size - 1
        cnt = buckets[lo]
        bar = "█" * cnt
        print(f"    {lo:4d}–{hi:4d} | {bar}  ({cnt})")
    print("-" * 70)

    return dict(short)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find ablation runs with fewer episodes than expected.")
    parser.add_argument("base_dir",  nargs="?", default="./test/333",
                        help="Root directory of the ablation experiment")
    parser.add_argument("--expected", type=int, default=None,
                        help="Expected number of episodes (default: auto = max found)")
    parser.add_argument("--target",   default="metrics.csv",
                        help="Filename suffix to search for (default: metrics.csv)")
    parser.add_argument("--filters",  nargs="*",
                        default=["M1", "M2", "M3", "M4",
                                 "Hidden", "Normal", "Out",
                                 "ALT", "CNT", "CRT", "DRT",
                                 "Small", "Medium", "Big"],
                        help="Filter names used to classify paths (same as SearchFilter keys)")

    args = parser.parse_args()

    short_runs = find_short_runs(
        base_dir     = args.base_dir,
        target_tail  = args.target,
        expected     = args.expected,
        filter_names = args.filters,
    )
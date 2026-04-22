#!/usr/bin/env python3
"""Find all metrics.csv files and sort by creation date."""

import os
import glob
from datetime import datetime

def find_metrics_by_date():
    """Find all metrics.csv files and display sorted by creation date."""

    # Find all metrics.csv files
    for root, dirs, files in os.walk('.'):
        for filename in files:
            if filename == 'metrics.csv':
                filepath = os.path.join(root, filename)
                creation_date = os.path.getctime(filepath)
                creation_date_str = datetime.fromtimestamp(creation_date).strftime('%Y-%m-%d %H:%M:%S')
                print(f"{creation_date_str}  {filepath}")

    # Sort by date
    print("\nSorted by creation date:")
    print("=" * 80)
    return sorted(glob.glob('**/metrics.csv', recursive=True), key=lambda x: os.path.getctime(x))

if __name__ == '__main__':
    for filepath in find_metrics_by_date():
        creation_date = os.path.getctime(filepath)
        creation_date_str = datetime.fromtimestamp(creation_date).strftime('%Y-%m-%d %H:%M:%S')
        relative_path = os.path.relpath(filepath)
        print(f"{creation_date_str}  {relative_path}")

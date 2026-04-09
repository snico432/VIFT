import csv
from pprint import pprint
import json
from pathlib import Path

def extract_metrics(file_path):
    with open(file_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        last_row = None
        for row in reader:
            last_row = row

    if last_row is None:
        return {}

    row_dict = dict(zip(header, last_row))

    seqs = [
        "test/05",
        "test/07",
        "test/10",
    ]

    errs = ["r_rel", "t_rel"]
    metrics = {}
    for seq in seqs:
        for err in errs:
            key = f"{seq}_{err}"
            val = row_dict.get(key, "").strip()
            metrics[key] = float(val) if val else None

    return metrics


if __name__ == "__main__":
    paths = {"alpha=10": "/home/ubuntu/cis-4910/VIFT/logs/train/runs/2026-04-01_22-33-30/csv/version_0",
             "alpha=20": "/home/ubuntu/cis-4910/VIFT/logs/train/runs/2026-04-05_22-15-15/csv/version_0",
             "alpha=23": "/home/ubuntu/cis-4910/VIFT/logs/train/runs/2026-04-09_00-10-11/csv/version_0",
             "alpha=25": "/home/ubuntu/cis-4910/VIFT/logs/train/runs/2026-04-08_23-31-53/csv/version_0",
             "alpha=27": "/home/ubuntu/cis-4910/VIFT/logs/train/runs/2026-04-08_03-20-19/csv/version_0",
             "alpha=30": "/home/ubuntu/cis-4910/VIFT/logs/train/runs/2026-04-05_23-23-06/csv/version_0", 
             "alpha=35": "/home/ubuntu/cis-4910/VIFT/logs/train/runs/2026-04-06_00-31-42/csv/version_0",
             "alpha=40": "/home/ubuntu/cis-4910/VIFT/logs/train/runs/2026-04-05_20-59-11/csv/version_0"}
    for alpha, path in paths.items():
        m = extract_metrics(Path(path) / "metrics.csv")
        print("-" * 100)
        print(alpha)
        pprint(m)
        print("-" * 100)

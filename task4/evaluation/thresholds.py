import json
import os
import numpy as np
from task4.evaluation.metrics import calibrate_threshold


def compute_all_thresholds(score_dict: dict, percentile: float = 95.0) -> dict:
    return {name: calibrate_threshold(scores, percentile)
            for name, scores in score_dict.items()}


def save_thresholds(thresholds: dict, method: str, results_dir: str):
    path = os.path.join(results_dir, f"{method}_thresholds.json")
    with open(path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"Thresholds saved → {path}")
    return path

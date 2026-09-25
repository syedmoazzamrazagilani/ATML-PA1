import os
import json
import numpy as np
from torchvision import datasets
from sklearn.model_selection import train_test_split

SEED       = 6304
VAL_SPLIT  = 0.1
DATA_DIR   = "data_cache"
OUT_PATH   = "task4/results/cifar10_splits_seed6304.json"


def make_splits():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    ds     = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    labels = np.array(ds.targets)
    idx    = np.arange(len(labels))

    train_idx, val_idx = train_test_split(
        idx, test_size=VAL_SPLIT, stratify=labels, random_state=SEED)

    splits = {
        "train_indices": train_idx.tolist(),
        "val_indices":   val_idx.tolist(),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Saved splits → {OUT_PATH}")
    print(f"  train={len(train_idx)}  val={len(val_idx)}")
    return splits


if __name__ == "__main__":
    make_splits()

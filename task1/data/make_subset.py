import os
import json
import torch
import numpy as np
from torchvision.datasets import STL10
from sklearn.model_selection import train_test_split
from task1.configs.config import SEED, DATA_DIR, NUM_TEST_SAMPLES

def prepare_splits():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    train_dataset = STL10(root=DATA_DIR, split='train', download=True)
    test_dataset = STL10(root=DATA_DIR, split='test', download=True)
    
    train_labels = train_dataset.labels
    test_labels = test_dataset.labels
    train_indices = np.arange(len(train_labels))
    test_indices = np.arange(len(test_labels))
    
    tr_idx, val_idx = train_test_split(
        train_indices,
        test_size=0.2,
        stratify=train_labels,
        random_state=SEED
    )
    
    selected_test_idx, _ = train_test_split(
        test_indices,
        train_size=NUM_TEST_SAMPLES,
        stratify=test_labels,
        random_state=SEED
    )
    
    splits = {
        "train_indices": tr_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_subset_indices": selected_test_idx.tolist()
    }
    
    split_path = os.path.join(DATA_DIR, "stl10_splits_seed6304.json")
    with open(split_path, "w") as f:
        json.dump(splits, f, indent=2)
        
    print(f"Splits saved to {split_path}")
    print(f"Train samples: {len(tr_idx)}, Val samples: {len(val_idx)}, Test subset: {len(selected_test_idx)}")

if __name__ == "__main__":
    prepare_splits()

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from task4.data.cifar10 import CIFAR10_MEAN, CIFAR10_STD

NEAR_CLASSES = [
    "bus", "pickup_truck", "motorcycle", "tractor",
    "wolf", "fox", "leopard", "camel",
]
FAR_CLASSES = [
    "bottle", "bowl", "chair", "clock",
    "keyboard", "mushroom", "sunflower", "wardrobe",
]


def _cifar100_fine_to_idx(class_names, fine_label_names):
    return [fine_label_names.index(c) for c in class_names]


def get_unknown_loaders(data_dir, batch_size=256, num_workers=2):
    """
    Returns (near_loader, far_loader, all_loader).
    Each yields (image_tensor, dummy_label=0) — labels are not used for scoring.
    """
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    c100 = datasets.CIFAR100(root=data_dir, train=False,
                              transform=eval_tf, download=True)

    fine_names = c100.classes   # 100 fine-grained class names, sorted

    near_indices_c100 = _cifar100_fine_to_idx(NEAR_CLASSES, fine_names)
    far_indices_c100  = _cifar100_fine_to_idx(FAR_CLASSES,  fine_names)

    targets = np.array(c100.targets)

    near_idx = np.where(np.isin(targets, near_indices_c100))[0]
    far_idx  = np.where(np.isin(targets, far_indices_c100))[0]
    all_idx  = np.concatenate([near_idx, far_idx])

    def _make_loader(idx):
        return DataLoader(Subset(c100, idx.tolist()),
                          batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)

    near_loader = _make_loader(near_idx)
    far_loader  = _make_loader(far_idx)
    all_loader  = _make_loader(all_idx)

    print(f"[CIFAR-100 unknowns] near={len(near_idx)}  far={len(far_idx)}  "
          f"total={len(all_idx)}")
    return near_loader, far_loader, all_loader


def get_unknown_class_labels(data_dir):
    """Returns per-image CIFAR-100 fine-class name for near and far sets."""
    c100 = datasets.CIFAR100(root=data_dir, train=False,
                              transform=transforms.ToTensor(), download=False)
    fine_names = c100.classes
    targets    = np.array(c100.targets)

    near_idx_c100 = _cifar100_fine_to_idx(NEAR_CLASSES, fine_names)
    far_idx_c100  = _cifar100_fine_to_idx(FAR_CLASSES,  fine_names)

    near_img_idx = np.where(np.isin(targets, near_idx_c100))[0]
    far_img_idx  = np.where(np.isin(targets, far_idx_c100))[0]

    near_class_names = [fine_names[targets[i]] for i in near_img_idx]
    far_class_names  = [fine_names[targets[i]] for i in far_img_idx]
    return near_class_names, far_class_names

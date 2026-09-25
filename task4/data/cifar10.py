import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_transforms(use_randaugment=False, ra_num_ops=2, ra_magnitude=9):
    """Return (train_transform, eval_transform)."""
    augment_ops = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if use_randaugment:
        augment_ops.append(
            transforms.RandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude)
        )
    augment_ops += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    train_tf = transforms.Compose(augment_ops)

    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_tf, eval_tf


def get_loaders(data_dir, batch_size=128, seed=6304,
                val_split=0.1, use_randaugment=False,
                ra_num_ops=2, ra_magnitude=9, num_workers=2):
    """
    Returns (train_loader, val_loader, test_loader).
    Stratified 90/10 split of the official CIFAR-10 training set.
    """
    train_tf, eval_tf = get_transforms(use_randaugment, ra_num_ops, ra_magnitude)

    full_train = datasets.CIFAR10(root=data_dir, train=True,
                                   transform=train_tf, download=True)
    full_val   = datasets.CIFAR10(root=data_dir, train=True,
                                   transform=eval_tf,  download=False)
    test_ds    = datasets.CIFAR10(root=data_dir, train=False,
                                   transform=eval_tf,  download=True)

    labels = np.array(full_train.targets)
    idx    = np.arange(len(labels))

    train_idx, val_idx = train_test_split(
        idx, test_size=val_split, stratify=labels, random_state=seed)

    train_loader = DataLoader(Subset(full_train, train_idx),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(Subset(full_val,   val_idx),
                              batch_size=256,       shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,
                              batch_size=256,       shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_idx, val_idx

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

def to_grayscale(img_tensor):
    gray = TF.rgb_to_grayscale(img_tensor, num_output_channels=3)
    return gray

def rotate_hue(img_tensor, factor=0.5):
    return TF.adjust_hue(img_tensor, factor)

def translate_image(img_tensor, shift_x, shift_y):
    # img_tensor: (C, H, W)
    pad = max(abs(shift_x), abs(shift_y))
    if pad == 0:
        return img_tensor
    padded = F.pad(img_tensor.unsqueeze(0), (pad, pad, pad, pad), mode='reflect').squeeze(0)
    _, h, w = img_tensor.shape
    start_y = pad + shift_y
    start_x = pad + shift_x
    return padded[:, start_y:start_y + h, start_x:start_x + w]

def shuffle_patches_4x4(img_tensor, seed=6304):
    # img_tensor: (C, 224, 224)
    c, h, w = img_tensor.shape
    grid_size = 4
    ph, pw = h // grid_size, w // grid_size
    
    patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            patch = img_tensor[:, i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            patches.append(patch)
            
    rng = np.random.RandomState(seed)
    perm = rng.permutation(grid_size * grid_size)
    
    shuffled_rows = []
    for row in range(grid_size):
        row_patches = [patches[perm[row * grid_size + col]] for col in range(grid_size)]
        shuffled_rows.append(torch.cat(row_patches, dim=2))
    return torch.cat(shuffled_rows, dim=1)

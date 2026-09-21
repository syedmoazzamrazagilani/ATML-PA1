import os
import random
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import STL10
from torchvision.utils import save_image
from task1.configs.config import SEED, DATA_DIR, STL10_CLASSES

torch.manual_seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())

def style_transfer(content_img, style_img, model, steps=300):
    target = content_img.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([target], lr=0.03)
    content_features = get_features(content_img, model)
    style_features = get_features(style_img, model)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    style_weights = {'conv1_1': 1.0, 'conv2_1': 0.8, 'conv3_1': 0.5, 'conv4_1': 0.3, 'conv5_1': 0.1}
    content_weight = 1e4
    style_weight = 1e9

    for _ in range(steps):
        target_features = get_features(target, model)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            style_loss += layer_style_loss / (d * h * w)
            
        total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
    return target.clamp(0, 1)

def generate_conflicts():
    print("Loading VGG19 for Style Transfer...")
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    for param in vgg.parameters():
        param.requires_grad = False

    train_ds = STL10(root=DATA_DIR, split='train', download=True)
    out_dir = os.path.join(DATA_DIR, "cue_conflicts")
    os.makedirs(out_dir, exist_ok=True)
    
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    class_pairs = list(itertools.combinations(random.sample(range(10), 5), 2))[:5]
    
    total = 0
    for (c_a, c_b) in class_pairs:
        imgs_a = [transform(img).unsqueeze(0).to(device) for img, lbl in train_ds if lbl == c_a][:20]
        imgs_b = [transform(img).unsqueeze(0).to(device) for img, lbl in train_ds if lbl == c_b][:20]
        
        for i in range(20):
            if total >= 200: break
            # A (Shape) + B (Texture)
            out1 = style_transfer(imgs_a[i], imgs_b[i], vgg, steps=200)
            save_image(out1, os.path.join(out_dir, f"shape_{c_a}_texture_{c_b}_{i}.png"))
            total += 1
            
            # B (Shape) + A (Texture)
            out2 = style_transfer(imgs_b[i], imgs_a[i], vgg, steps=200)
            save_image(out2, os.path.join(out_dir, f"shape_{c_b}_texture_{c_a}_{i}.png"))
            total += 1
            print(f"Generated {total}/200 conflicts")

if __name__ == "__main__":
    generate_conflicts()

import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
import torchvision.transforms as T

from task1.configs.config import SEED, DATA_DIR, RESULTS_DIR, BATCH_SIZE, STL10_CLASSES
from task1.models.backbones import ModelWrapper
from task1.data.transforms import to_grayscale, rotate_hue, translate_image, shuffle_patches_4x4

def calculate_shape_bias(n_shape, n_texture, n_total):
    """Calculates Shape Bias(%) and Coverage(%) as defined in the manual."""
    if (n_shape + n_texture) == 0: 
        return 0.0, 0.0
    shape_bias = (n_shape / (n_shape + n_texture)) * 100
    coverage = ((n_shape + n_texture) / n_total) * 100
    return shape_bias, coverage

def run_evaluations():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(DATA_DIR, "stl10_splits_seed6304.json"), "r") as f:
        splits = json.load(f)
        
    models_to_test = ["resnet50", "vit_b_16", "clip_vit_b_32"]
    final_results = {}
    
    pre_norm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    test_ds = STL10(root=DATA_DIR, split='test', transform=pre_norm, download=False)
    test_loader = DataLoader(Subset(test_ds, splits["test_subset_indices"]), batch_size=BATCH_SIZE, shuffle=False)
    
    for m_name in models_to_test:
        print(f"\n--- Running Interventions on {m_name} ---")
        if m_name == "clip_vit_b_32":
            norm = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        else:
            norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        wrapper = ModelWrapper(model_name=m_name, num_classes=len(STL10_CLASSES)).to(device)
        
        print("Quickly reloading linear head weights...")
        train_ds = STL10(root=DATA_DIR, split='train', transform=T.Compose([pre_norm, norm]), download=False)
        train_loader = DataLoader(Subset(train_ds, splits["train_indices"]), batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.AdamW(wrapper.head.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        wrapper.train()
        for epoch in range(12): 
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits, _ = wrapper(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
        wrapper.eval()
        
        m_res = {}
        
        # 1. Color and Patch Interventions
        interventions = {
            "grayscale": lambda x: to_grayscale(x),
            "hue_rotation": lambda x: rotate_hue(x, factor=0.5),
            "patch_shuffle": lambda x: shuffle_patches_4x4(x, seed=SEED)
        }
        
        for inv_name, inv_func in interventions.items():
            preds_list, labels_list = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x_inv = torch.stack([inv_func(img) for img in x])
                    x_norm = torch.stack([norm(img) for img in x_inv]).to(device)
                    logits, _ = wrapper(x_norm)
                    preds_list.extend(logits.argmax(dim=1).cpu().numpy())
                    labels_list.extend(y.numpy())
            
            acc = float((np.array(preds_list) == np.array(labels_list)).mean())
            m_res[inv_name] = acc
            print(f"[{inv_name}] Accuracy: {acc:.4f}")
            
        # 2. Translation (Average over 4 directions: +x, -x, +y, -y)
        shifts = [8, 16, 32]
        for shift in shifts:
            shift_accs = []
            directions = [(shift, 0), (-shift, 0), (0, shift), (0, -shift)]
            for dx, dy in directions:
                preds_list, labels_list = [], []
                with torch.no_grad():
                    for x, y in test_loader:
                        x_inv = torch.stack([translate_image(img, dx, dy) for img in x])
                        x_norm = torch.stack([norm(img) for img in x_inv]).to(device)
                        logits, _ = wrapper(x_norm)
                        preds_list.extend(logits.argmax(dim=1).cpu().numpy())
                        labels_list.extend(y.numpy())
                
                shift_accs.append((np.array(preds_list) == np.array(labels_list)).mean())
            
            avg_acc = float(np.mean(shift_accs))
            m_res[f"translation_{shift}px"] = avg_acc
            print(f"[translation_{shift}px] Avg Accuracy: {avg_acc:.4f}")
            
        # 3. Cue Conflicts (Shape vs Texture)
        cue_dir = os.path.join(DATA_DIR, "cue_conflicts")
        if os.path.exists(cue_dir):
            n_shape, n_texture, n_total = 0, 0, 0
            with torch.no_grad():
                for fname in os.listdir(cue_dir):
                    if not fname.endswith(".png"): continue
                    # Format generated earlier: shape_3_texture_5_0.png
                    parts = fname.split("_")
                    shape_class = int(parts[1])
                    texture_class = int(parts[3])
                    
                    img = Image.open(os.path.join(cue_dir, fname)).convert("RGB")
                    x = pre_norm(img)
                    x_norm = norm(x).unsqueeze(0).to(device)
                    
                    logits, _ = wrapper(x_norm)
                    pred = logits.argmax(dim=1).item()
                    
                    if pred == shape_class:
                        n_shape += 1
                    elif pred == texture_class:
                        n_texture += 1
                    n_total += 1
                    
            sb, cov = calculate_shape_bias(n_shape, n_texture, n_total)
            m_res["shape_bias_pct"] = float(sb)
            m_res["coverage_pct"] = float(cov)
            print(f"[Cue Conflict] Shape Bias: {sb:.2f}%, Coverage: {cov:.2f}%")
        else:
            print("[Cue Conflict] Directory not found. Did the generation script finish?")
            
        final_results[m_name] = m_res
        
    with open(os.path.join(RESULTS_DIR, "bias_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved all bias results to {RESULTS_DIR}/bias_results.json")

if __name__ == "__main__":
    run_evaluations()

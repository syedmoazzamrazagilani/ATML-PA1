import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import STL10
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import f1_score
import open_clip

from task1.configs.config import SEED, DATA_DIR, RESULTS_DIR, BATCH_SIZE, LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE, STL10_CLASSES
from task1.models.backbones import ModelWrapper

def get_base_transform(model_name):
    # Common 224x224 resize and crop before applying model-specific normalization
    base = [T.Resize(256), T.CenterCrop(224), T.ToTensor()]
    
    if model_name == "clip_vit_b_32":
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
    base.append(T.Normalize(mean=mean, std=std))
    return T.Compose(base)

def train_and_evaluate():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    with open(os.path.join(DATA_DIR, "stl10_splits_seed6304.json"), "r") as f:
        splits = json.load(f)
        
    models_to_test = ["resnet50", "vit_b_16", "clip_vit_b_32"]
    results = {}
    
    for m_name in models_to_test:
        print(f"\n--- Processing {m_name} ---")
        transform = get_base_transform(m_name)
        
        train_ds = STL10(root=DATA_DIR, split='train', transform=transform, download=False)
        test_ds = STL10(root=DATA_DIR, split='test', transform=transform, download=False)
        
        train_loader = DataLoader(Subset(train_ds, splits["train_indices"]), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(train_ds, splits["val_indices"]), batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(Subset(test_ds, splits["test_subset_indices"]), batch_size=BATCH_SIZE, shuffle=False)
        
        wrapper = ModelWrapper(model_name=m_name, num_classes=len(STL10_CLASSES)).to(device)
        
        # 1. Train linear head using AdamW and Early Stopping
        optimizer = torch.optim.AdamW(wrapper.head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        patience_counter = 0
        best_weights = wrapper.head.state_dict()
        
        for epoch in range(MAX_EPOCHS):
            wrapper.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits, _ = wrapper(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                
            wrapper.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits, _ = wrapper(x)
                    val_correct += (logits.argmax(dim=1) == y).sum().item()
                    val_total += y.size(0)
                    
            val_acc = val_correct / val_total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_weights = wrapper.head.state_dict()
            else:
                patience_counter += 1
                
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        wrapper.head.load_state_dict(best_weights)
        
        # 2. Evaluate Clean Baseline
        wrapper.eval()
        all_preds, all_labels, all_confs = [], [], []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = wrapper(x)
                probs = logits.softmax(dim=-1)
                confs, preds = probs.max(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_confs.extend(confs.cpu().numpy())
                
        acc = float((np.array(all_preds) == np.array(all_labels)).mean())
        macro_f1 = float(f1_score(all_labels, all_preds, average='macro'))
        mean_conf = float(np.mean(all_confs))
        results[f"{m_name}_linear"] = {"acc": acc, "f1": macro_f1, "conf": mean_conf}
        print(f"{m_name} Linear Head -> Acc: {acc:.4f}, F1: {macro_f1:.4f}, Conf: {mean_conf:.4f}")
        
        # 3. Evaluate Zero-shot CLIP
        if m_name == "clip_vit_b_32":
            print("Evaluating Zero-Shot CLIP...")
            text_prompts = [f"a photo of a {c}." for c in STL10_CLASSES]
            text_tokens = open_clip.tokenize(text_prompts).to(device)
            
            zs_preds, zs_confs = [], []
            with torch.no_grad():
                text_features = wrapper.full_clip.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                for x, y in test_loader:
                    x = x.to(device)
                    image_features = wrapper.full_clip.encode_image(x)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    logits = (100.0 * image_features @ text_features.T)
                    probs = logits.softmax(dim=-1)
                    confs, preds = probs.max(dim=1)
                    
                    zs_preds.extend(preds.cpu().numpy())
                    zs_confs.extend(confs.cpu().numpy())
                    
            zs_acc = float((np.array(zs_preds) == np.array(all_labels)).mean())
            zs_macro_f1 = float(f1_score(all_labels, zs_preds, average='macro'))
            zs_mean_conf = float(np.mean(zs_confs))
            results["clip_zeroshot"] = {"acc": zs_acc, "f1": zs_macro_f1, "conf": zs_mean_conf}
            print(f"CLIP Zero-Shot -> Acc: {zs_acc:.4f}, F1: {zs_macro_f1:.4f}, Conf: {zs_mean_conf:.4f}")
            
    with open(os.path.join(RESULTS_DIR, "clean_baseline.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Clean baseline results saved to {RESULTS_DIR}/clean_baseline.json")

if __name__ == "__main__":
    train_and_evaluate()

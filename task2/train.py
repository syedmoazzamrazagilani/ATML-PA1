import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import numpy as np
from sklearn.metrics import f1_score
import itertools

from shared.pacs_protocol import get_transforms
from task2.models.backbone import ResNet18Backbone, ClassifierHead
from task2.models.domain_discriminator import DomainDiscriminator, get_alpha_schedule
from task2.methods.source_only import SourceOnlyLoss
from task2.methods.dan import MMDLoss
from task2.methods.cdan import multilinear_conditioning
from task2.methods.dann import DANNLoss

def load_config(method):
    with open("task2/configs/base.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open(f"task2/configs/{method}.yaml", "r") as f:
        method_config = yaml.safe_load(f)
    config.update(method_config)
    return config

def train(method):
    config = load_config(method)
    torch.manual_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(config["checkpoints_dir"], exist_ok=True)
    
    train_tf, eval_tf = get_transforms()
    import json
    with open(config["splits_path"], "r") as f:
        splits = json.load(f)
        
    source_loaders = []
    val_loaders = {}
    
    for domain in config["source_domains"]:
        ds_train = datasets.ImageFolder(os.path.join(config["data_dir"], domain), transform=train_tf)
        ds_val = datasets.ImageFolder(os.path.join(config["data_dir"], domain), transform=eval_tf)
        
        train_sub = Subset(ds_train, splits[domain]["train"])
        val_sub = Subset(ds_val, splits[domain]["val"])
        
        source_loaders.append(DataLoader(train_sub, batch_size=config["batch_size_per_source"], shuffle=True, drop_last=True))
        val_loaders[domain] = DataLoader(val_sub, batch_size=32, shuffle=False)
        
    ds_target = datasets.ImageFolder(os.path.join(config["data_dir"], config["target_domain"]), transform=train_tf)
    target_loader = DataLoader(ds_target, batch_size=config["target_batch_size"], shuffle=True, drop_last=True)
    
    backbone = ResNet18Backbone().to(device)
    classifier = ClassifierHead(num_classes=config["num_classes"]).to(device)
    
    params = list(backbone.parameters()) + list(classifier.parameters())
    
    domain_disc = None
    if method == "dann":
        domain_disc = DomainDiscriminator(in_features=512).to(device)
        params += list(domain_disc.parameters())
    elif method == "cdan":
        domain_disc = DomainDiscriminator(in_features=3584).to(device)
        params += list(domain_disc.parameters())
        
    optimizer = torch.optim.AdamW(params, lr=config["lr"], weight_decay=config["weight_decay"])
    class_criterion = nn.CrossEntropyLoss()
    
    mmd_criterion = MMDLoss() if method == "dan" else None
    domain_criterion = DANNLoss() if method in ["dann", "cdan"] else None
    
    best_mean_f1 = 0.0
    patience_counter = 0
    
    total_batches = min([len(l) for l in source_loaders])
    
    for epoch in range(config["max_epochs"]):
        backbone.train() 
        classifier.train()
        if domain_disc:
            domain_disc.train()
            
        target_iter = itertools.cycle(target_loader)
        source_iters = [iter(l) for l in source_loaders]
        
        for batch_idx in range(total_batches):
            p = (epoch * total_batches + batch_idx) / (config["max_epochs"] * total_batches)
            alpha = get_alpha_schedule(p)
            
            source_x, source_y = [], []
            for s_iter in source_iters:
                x, y = next(s_iter)
                source_x.append(x)
                source_y.append(y)
            
            source_x = torch.cat(source_x, dim=0).to(device)
            source_y = torch.cat(source_y, dim=0).to(device)
            
            target_x, _ = next(target_iter)
            target_x = target_x.to(device)
            
            optimizer.zero_grad()
            
            source_feat = backbone(source_x)
            source_logits = classifier(source_feat)
            cls_loss = class_criterion(source_logits, source_y)
            
            loss = cls_loss
            
            if method != "source_only":
                target_feat = backbone(target_x)
                
                if method == "dan":
                    loss += config["lambda_mmd"] * mmd_criterion(source_feat, target_feat)
                    
                elif method in ["dann", "cdan"]:
                    target_logits = classifier(target_feat)
                    
                    if method == "cdan":
                        s_g = multilinear_conditioning(source_feat, source_logits)
                        t_g = multilinear_conditioning(target_feat, target_logits)
                    else:
                        s_g = source_feat
                        t_g = target_feat
                        
                    d_labels = torch.cat([
                        torch.ones(source_x.size(0), dtype=torch.long),
                        torch.zeros(target_x.size(0), dtype=torch.long)
                    ]).to(device)
                    
                    d_logits = domain_disc(torch.cat([s_g, t_g], dim=0), alpha)
                    loss += config["loss_weight"] * domain_criterion(d_logits, d_labels)
                    
            loss.backward()
            optimizer.step()
            
        backbone.eval()
        classifier.eval()
        
        domain_f1s = []
        with torch.no_grad():
            for domain, v_loader in val_loaders.items():
                preds, labels = [], []
                for x, y in v_loader:
                    x = x.to(device)
                    logits = classifier(backbone(x))
                    preds.extend(logits.argmax(dim=1).cpu().numpy())
                    labels.extend(y.numpy())
                
                macro_f1 = f1_score(labels, preds, average='macro')
                domain_f1s.append(macro_f1)
                
        mean_f1 = np.mean(domain_f1s)
        print(f"Epoch {epoch+1} | Mean Source F1: {mean_f1:.4f}")
        
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            patience_counter = 0
            
            checkpoint = {
                "backbone": backbone.state_dict(),
                "classifier": classifier.state_dict()
            }
            torch.save(checkpoint, os.path.join(config["checkpoints_dir"], f"{method}_best.pth"))
        else:
            patience_counter += 1
            
        if patience_counter >= config["patience"]:
            print(f"Early stopping at epoch {epoch+1}")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["source_only", "dan", "dann", "cdan"])
    args = parser.parse_args()
    train(args.method)

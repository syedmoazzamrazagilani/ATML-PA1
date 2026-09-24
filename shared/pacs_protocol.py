import os
import json
import shutil
import urllib.request
import zipfile
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

def get_transforms():
    # Required: 256x256 resize, 224x224 random crop, horizontal flip for training
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Required: 256x256 resize, 224x224 center crop for validation
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, eval_transform

def download_and_extract_pacs(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    photo_dir = os.path.join(data_dir, "photo")
    
    if not os.path.exists(photo_dir):
        print("Downloading PACS dataset...")
        url = "https://wjdcloud.blob.core.windows.net/dataset/PACS.zip"
        zip_path = os.path.join(data_dir, "PACS.zip")
        urllib.request.urlretrieve(url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)
        
        kfold_dir = os.path.join(data_dir, "kfold")
        if os.path.exists(kfold_dir):
            for dom in ["photo", "art_painting", "cartoon", "sketch"]:
                shutil.move(os.path.join(kfold_dir, dom), os.path.join(data_dir, dom))
            os.rmdir(kfold_dir)
            
def create_pacs_splits(data_dir, split_dir, seed=6304):
    download_and_extract_pacs(data_dir)
    source_domains = ["photo", "art_painting", "cartoon"]
    
    splits = {}
    for domain in source_domains:
        domain_path = os.path.join(data_dir, domain)
        dataset = ImageFolder(root=domain_path)
        labels = dataset.targets
        indices = list(range(len(labels)))
        
        # 80/20 Stratified Split
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=seed
        )
        splits[domain] = {"train": train_idx, "val": val_idx}
        
    os.makedirs(split_dir, exist_ok=True)
    split_file = os.path.join(split_dir, f"pacs_sketch_seed{seed}.json")
    with open(split_file, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits saved to {split_file}")

if __name__ == "__main__":
    create_pacs_splits(data_dir="data_cache/PACS", split_dir="shared/splits", seed=6304)

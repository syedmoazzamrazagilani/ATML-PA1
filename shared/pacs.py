import os
from torchvision.datasets import ImageFolder

def get_pacs_domain(data_dir, domain, transform=None):
    """Loads a specific domain from the PACS dataset."""
    domain_path = os.path.join(data_dir, domain)
    if not os.path.exists(domain_path):
        raise RuntimeError(f"Domain path {domain_path} does not exist. Run protocol script first.")
    return ImageFolder(root=domain_path, transform=transform)

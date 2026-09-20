import os

SEED = 6304
DATASET = "STL10"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data_cache")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Training & Eval Hyperparameters
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
PATIENCE = 5
NUM_TEST_SAMPLES = 500  # 50 per class for STL-10

STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck"
]

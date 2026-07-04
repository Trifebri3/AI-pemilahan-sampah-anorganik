import os

# Hyperparameters
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 10
INIT_LR = 1e-3
NUM_CLASSES = 5
CLASSES = ["Kaca", "Kertas", "Logam", "Plastik", "Residu"]

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_split")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
HISTORY_SAVE_DIR = os.path.join(BASE_DIR, "history")
VISUALIZATION_DIR = os.path.join(BASE_DIR, "visualizations")

# Ensure folders exist
for folder in [MODEL_SAVE_DIR, HISTORY_SAVE_DIR, VISUALIZATION_DIR]:
    os.makedirs(folder, exist_ok=True)

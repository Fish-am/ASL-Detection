from pathlib import Path

class Config:
    # Data paths
    DATA_ROOT = Path('data')
    VIDEO_DIR = DATA_ROOT / 'videos'
    JSON_PATH = DATA_ROOT / 'WLASL_v0.3.json'
    CHECKPOINT_DIR = Path('checkpoints')
    
    # Training parameters
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 0.001
    IMG_SIZE = 640
    NUM_WORKERS = 4
    
    # Model parameters
    PRETRAINED = True
    DEVICE = 'cuda'  # or 'cpu'
    
    # Data augmentation parameters
    TRAIN_SPLIT = 0.8
    RANDOM_SEED = 42
    
    # Create required directories
    DATA_ROOT.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True) 
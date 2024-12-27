from pathlib import Path

class Config:
    
    DATA_ROOT = Path('data')
    VIDEO_DIR = DATA_ROOT / 'videos'
    JSON_PATH = DATA_ROOT / 'WLASL_v0.3.json'
    CHECKPOINT_DIR = Path('checkpoints')
    
    
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 0.001
    IMG_SIZE = 640
    NUM_WORKERS = 4
    
    
    PRETRAINED = True
    DEVICE = 'cuda' 
    
    
    TRAIN_SPLIT = 0.8
    RANDOM_SEED = 42
    
    
    DATA_ROOT.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True) 
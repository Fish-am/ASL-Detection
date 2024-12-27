import torch
import random
import numpy as np
from pathlib import Path
import json

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_label_map(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)['root']
    return {item['gloss']: idx for idx, item in enumerate(data)
            if item['instances'] is not None}

def create_data_splits(dataset, train_ratio=0.8, seed=42):
    """Split dataset into train and validation sets"""
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(train_ratio * dataset_size))
    
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    return train_indices, val_indices

def save_checkpoint(state, is_best, checkpoint_dir):
    """Save model checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    
    filepath = checkpoint_dir / 'last_checkpoint.pt'
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = checkpoint_dir / 'best_model.pt'
        torch.save(state, best_filepath) 
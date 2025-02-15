import torch
from torch.utils.data import DataLoader
from dataset import WLASLDataset
from model import YOLOv10ASL
import torch.optim as optim
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transforms import get_training_transforms, get_validation_transforms
from loss import YOLOLoss
from utils import create_data_splits
from config import Config
from tqdm import tqdm

def collate_fn(batch):
    images = []
    labels = []
    for img, lbl in batch:
        images.append(img)
        labels.append(lbl)
    images = torch.stack(images)
    labels = torch.cat(labels, dim=0).reshape(len(batch), 1, -1)  
    return images, labels

def train():
    
    BATCH_SIZE = 2
    EPOCHS = 100
    LEARNING_RATE = 0.0001
    IMG_SIZE = 416      
    GRID_SIZE = 13      
    
    
    json_path = Config.JSON_PATH
    video_dir = Config.VIDEO_DIR
    
    
    train_transform = get_training_transforms(IMG_SIZE)
    val_transform = get_validation_transforms(IMG_SIZE)
    
    
    full_dataset = WLASLDataset(json_path, video_dir, img_size=IMG_SIZE)
    train_indices, val_indices = create_data_splits(full_dataset, Config.TRAIN_SPLIT, Config.RANDOM_SEED)
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=True
    )
    
   
    model = YOLOv10ASL(num_classes=len(full_dataset.label_map), grid_size=GRID_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    criterion = YOLOLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    best_val_loss = float('inf')
    
   
    max_norm = 1.0
    
    
    torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(EPOCHS):
        
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(device)
            targets = targets.to(device)
            
            
            if batch_idx == 0:
                print(f"\nDebug shapes:")
                print(f"Images shape: {images.shape}")
                print(f"Targets shape: {targets.shape}")
            
            optimizer.zero_grad()
            predictions = model(images)
            
            if batch_idx == 0:
                print(f"Predictions shape: {predictions.shape}\n")
            
            loss = criterion(predictions, targets)
            
            loss.backward()
            
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            
            del images, targets, predictions
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        avg_train_loss = train_loss / len(train_loader)
        
        
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                predictions = model(images)
                loss = criterion(predictions, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}")
        
       
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'checkpoints/best_model.pt')

if __name__ == '__main__':
    
    Path('checkpoints').mkdir(exist_ok=True)
    train() 
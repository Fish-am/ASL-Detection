import json
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WLASLDataset(Dataset):
    def __init__(self, json_path, video_dir, img_size=640, transform=None):
        self.video_dir = Path(video_dir)
        self.img_size = img_size
        self.transform = transform
        
        # Get available videos first
        available_videos = set(f.stem for f in self.video_dir.glob('*.mp4'))
        
        # Load JSON data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        # Create label mapping starting from 0
        unique_classes = sorted(set(item['gloss'] for item in self.data 
                                 if item['instances'] is not None))
        self.label_map = {gloss: idx for idx, gloss in enumerate(unique_classes)}
        
        # Create instances list
        self.instances = []
        for item in self.data:
            if item['instances'] is not None:
                valid_instances = [
                    inst for inst in item['instances']
                    if inst['video_id'] in available_videos
                ]
                if valid_instances:
                    for instance in valid_instances:
                        self.instances.append({
                            'video_id': instance['video_id'],
                            'label': self.label_map[item['gloss']],  # Use mapped index
                            'gloss': item['gloss']
                        })
        
        logger.info(f"Found {len(self.instances)} valid instances out of {len(available_videos)} available videos")
        logger.info(f"Number of classes: {len(self.label_map)}")

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        video_path = self.video_dir / f"{instance['video_id']}.mp4"
        
        # Get original instance data for bbox
        original_instance = None
        for item in self.data:
            if item['instances'] is not None:
                for inst in item['instances']:
                    if inst['video_id'] == instance['video_id']:
                        original_instance = inst
                        break
            if original_instance:
                break
        
        # Extract center frame from video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use frame_start and frame_end if available
        if original_instance and 'frame_end' in original_instance and original_instance['frame_end'] != -1:
            target_frame = (original_instance['frame_start'] + original_instance['frame_end']) // 2
        else:
            target_frame = frame_count // 2
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame from video: {video_path}")
        
        # Get original frame dimensions
        orig_h, orig_w = frame.shape[:2]
        
        # Extract bbox coordinates and normalize
        if original_instance and 'bbox' in original_instance:
            bbox = original_instance['bbox']
        else:
            # If no bbox, use full frame
            bbox = [0, 0, orig_w, orig_h]
            
        # Convert bbox to YOLO format (x_center, y_center, width, height)
        x_center = ((bbox[0] + bbox[2]) / 2) / orig_w
        y_center = ((bbox[1] + bbox[3]) / 2) / orig_h
        width = (bbox[2] - bbox[0]) / orig_w
        height = (bbox[3] - bbox[1]) / orig_h
        
        # Resize frame
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create YOLO format labels
        label = torch.zeros((1, 6))  # class, x, y, w, h, confidence
        label[0] = torch.tensor([
            instance['label'],  # class
            x_center,          # x
            y_center,          # y
            width,            # w
            height,           # h
            1.0               # confidence
        ])
        
        # Ensure label is on the same device as the image
        if self.transform:
            transformed = self.transform(image=frame)
            frame = transformed['image']
        else:
            frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        
        return frame, label 
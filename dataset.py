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
        
        
        available_videos = set(f.stem for f in self.video_dir.glob('*.mp4'))
        
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        
        unique_classes = sorted(set(item['gloss'] for item in self.data 
                                 if item['instances'] is not None))
        self.label_map = {gloss: idx for idx, gloss in enumerate(unique_classes)}
        
        
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
                            'label': self.label_map[item['gloss']],  
                            'gloss': item['gloss']
                        })
        
        logger.info(f"Found {len(self.instances)} valid instances out of {len(available_videos)} available videos")
        logger.info(f"Number of classes: {len(self.label_map)}")

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        instance = self.instances[idx]
        video_path = self.video_dir / f"{instance['video_id']}.mp4"
        
        
        original_instance = None
        for item in self.data:
            if item['instances'] is not None:
                for inst in item['instances']:
                    if inst['video_id'] == instance['video_id']:
                        original_instance = inst
                        break
            if original_instance:
                break
        
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        
        if original_instance and 'frame_end' in original_instance and original_instance['frame_end'] != -1:
            target_frame = (original_instance['frame_start'] + original_instance['frame_end']) // 2
        else:
            target_frame = frame_count // 2
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame from video: {video_path}")
        
        
        orig_h, orig_w = frame.shape[:2]
        
        
        if original_instance and 'bbox' in original_instance:
            bbox = original_instance['bbox']
        else:
            
            bbox = [0, 0, orig_w, orig_h]
            
        
        x_center = ((bbox[0] + bbox[2]) / 2) / orig_w
        y_center = ((bbox[1] + bbox[3]) / 2) / orig_h
        width = (bbox[2] - bbox[0]) / orig_w
        height = (bbox[3] - bbox[1]) / orig_h
        
        
        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        
        label = torch.zeros((1, 6))  
        label[0] = torch.tensor([
            instance['label'],  
            x_center,          
            y_center,          
            width,            
            height,           
            1.0               
        ])
        
        
        if self.transform:
            transformed = self.transform(image=frame)
            frame = transformed['image']
        else:
            frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
        
        return frame, label 
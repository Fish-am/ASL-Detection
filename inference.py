import torch
import cv2
import numpy as np
from pathlib import Path
from model import YOLOv10ASL
from transforms import get_validation_transforms
from config import Config

class ASLPredictor:
    def __init__(self, model_path, num_classes):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLOv10ASL(num_classes=num_classes)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = get_validation_transforms(Config.IMG_SIZE)
    
    def predict_video(self, video_path, confidence_threshold=0.5):
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        predictions = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            transformed = self.transform(image=frame_rgb)
            image = transformed['image'].unsqueeze(0).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                output = self.model(image)
            
            # Process predictions
            pred = output[0].cpu().numpy()
            confident_preds = pred[pred[:, 4] > confidence_threshold]
            predictions.append(confident_preds)
        
        cap.release()
        return predictions

def main():
    # Example usage
    model_path = 'checkpoints/best_model.pt'
    video_path = 'test_videos/example.mp4'
    num_classes = 2000  # Update based on your dataset
    
    predictor = ASLPredictor(model_path, num_classes)
    predictions = predictor.predict_video(video_path)
    
    print("Predictions:", predictions)

if __name__ == '__main__':
    main() 
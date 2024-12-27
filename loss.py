import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5, grid_size=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.grid_size = grid_size
        
    def forward(self, predictions, targets):
        # Predictions: (batch_size, grid_size, grid_size, num_classes + 5)
        # Targets: (batch_size, max_objects, 6) [class, x, y, w, h, conf]
        
        obj_mask = targets[..., 5] == 1
        noobj_mask = targets[..., 5] == 0
        
        # Convert target coordinates to grid cell coordinates
        cell_size = 1.0 / self.grid_size
        targets[..., 1:3] = targets[..., 1:3] / cell_size  # x, y
        targets[..., 3:5] = targets[..., 3:5] / cell_size  # w, h
        
        # Coordinate loss
        coord_loss = self.mse(
            predictions[..., 1:5][obj_mask],
            targets[..., 1:5][obj_mask]
        )
        
        # Object confidence loss
        obj_conf_loss = self.bce(
            predictions[..., 0][obj_mask],
            targets[..., 5][obj_mask]
        )
        
        # No object confidence loss
        noobj_conf_loss = self.bce(
            predictions[..., 0][noobj_mask],
            targets[..., 5][noobj_mask]
        )
        
        # Class prediction loss
        class_loss = self.bce(
            predictions[..., 5:][obj_mask],
            targets[..., 0][obj_mask]
        )
        
        total_loss = (
            self.lambda_coord * coord_loss +
            obj_conf_loss +
            self.lambda_noobj * noobj_conf_loss +
            class_loss
        )
        
        return total_loss 
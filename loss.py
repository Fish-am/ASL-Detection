import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5, grid_size=13):
        super().__init__()
        self.mse = nn.MSELoss(reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.grid_size = grid_size
        
    def forward(self, predictions, targets):
        batch_size = predictions.size(0)
        num_classes = predictions.size(-1) - 5
        
        
        targets = targets.repeat(1, self.grid_size * self.grid_size, 1)
        
        
        pred_boxes = predictions[..., :5].reshape(batch_size, -1, 5)
        pred_classes = predictions[..., 5:].reshape(batch_size, -1, num_classes)
        
       
        obj_mask = (targets[..., 5] > 0).float()
        noobj_mask = (targets[..., 5] == 0).float()
        
        
        box_loss = self.mse(
            pred_boxes[..., 1:5] * obj_mask.unsqueeze(-1),
            targets[..., 1:5] * obj_mask.unsqueeze(-1)
        )
        
        
        conf_obj_loss = self.bce(
            pred_boxes[..., 0],
            targets[..., 5]
        )
        
       
        class_targets = torch.clamp(targets[..., 0], 0, num_classes - 1).long()
        one_hot_targets = torch.nn.functional.one_hot(class_targets, num_classes)
        class_loss = self.bce(
            pred_classes,
            one_hot_targets.float()
        )
        
        
        total_loss = (
            self.lambda_coord * box_loss +
            conf_obj_loss +
            class_loss
        )
        
        return total_loss 
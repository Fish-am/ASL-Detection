import torch
import torch.nn as nn

class YOLOv10ASL(nn.Module):
    def __init__(self, num_classes, grid_size=20):
        super(YOLOv10ASL, self).__init__()
        self.grid_size = grid_size
        
        # Backbone
        self.backbone = nn.Sequential(
            self._conv_block(3, 32, 3),
            self._conv_block(32, 64, 3),
            nn.MaxPool2d(2),
            self._conv_block(64, 128, 3),
            self._conv_block(128, 256, 3),
            nn.MaxPool2d(2),
            self._conv_block(256, 512, 3),
            self._conv_block(512, 1024, 3),
            nn.MaxPool2d(2),
        )
        
        # Detection head
        self.head = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, (num_classes + 5), 1)  # 5 for x, y, w, h, confidence
        )
    
    def _conv_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)
        x = self.head(x)
        # Reshape to (batch_size, grid_size, grid_size, num_classes + 5)
        return x.view(batch_size, self.grid_size, self.grid_size, -1) 
import torch
import torch.nn as nn
from torchinfo import summary
from models.layers import DepthwiseSeparableConv

class SmallMobile(nn.Module):
    
    def __init__(self, in_channels : int = 3, num_class : int = 8):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            DepthwiseSeparableConv(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            DepthwiseSeparableConv(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            DepthwiseSeparableConv(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
            DepthwiseSeparableConv(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.head = nn.Linear(in_features=128, out_features=num_class)
        
        
    def forward(self, x):
        
        x = self.model(x)
        x = x.flatten(1)
        
        return self.head(x)
    
    
if __name__ == "__main__":
    model = SmallMobile()
    print(summary(model, input_size=(1, 3, 224, 224), depth=2))
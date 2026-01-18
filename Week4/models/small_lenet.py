import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class SmallLeNet(nn.Module):
    
    def __init__(self, in_channels : int = 3, num_class : int = 8):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1,1)),
        )
        
        self.head = nn.Linear(in_features=64, out_features=8)
        
        
    def forward(self, x):
        
        x = self.model(x)
        x = x.flatten(1)
        
        return self.head(x)
    
    
    
    
if __name__ == "__main__":
    model = SmallLeNet()
    print(summary(model, input_data=torch.randn((1, 3, 224, 224))))
import torch
import torch.nn as nn
from torchinfo import summary
from models.layers import DepthwiseSeparableConv, ResidualBlock, ResidualBlockDepthwise

class SmallResnet(nn.Module):
    
    def __init__(self, in_channels : int = 3, num_class : int = 8):
        super().__init__()
        
        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            ResidualBlock(channels=64, kernel_size=3),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            ResidualBlock(channels=128, kernel_size=3),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            
            ResidualBlock(channels=256, kernel_size=3),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            
            ResidualBlock(channels=512, kernel_size=3),
            
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Flatten()
        )
        
        self.head = nn.Sequential(
            nn.Linear(in_features=512 * 4 * 4, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=num_class)
        )
        
        
    def forward(self, x):
        
        return self.head(self.model(x))    

class SmallResnetDepthwise(nn.Module):
    
    def __init__(self, in_channels : int = 3, num_class : int = 8):
        super().__init__()
        
        self.model = nn.Sequential(
            
            DepthwiseSeparableConv(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            ResidualBlockDepthwise(channels=64, kernel_size=3),
            
            DepthwiseSeparableConv(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            
            ResidualBlockDepthwise(channels=128, kernel_size=3),
            
            DepthwiseSeparableConv(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            
            ResidualBlockDepthwise(channels=256, kernel_size=3),
            
            DepthwiseSeparableConv(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            
            ResidualBlockDepthwise(channels=512, kernel_size=3),
            
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Flatten()
        )
        
        self.head = nn.Sequential(
            nn.Linear(in_features=512 * 4 * 4, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=num_class)
        )
        
        
    def forward(self, x):
        
        return self.head(self.model(x))
    
class SmallResnetReduce(nn.Module):
    
    def __init__(self, in_channels : int = 3, num_class : int = 8):
        super().__init__()
        
        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            ResidualBlock(channels=64, kernel_size=3),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            ResidualBlock(channels=128, kernel_size=3),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            
            ResidualBlock(channels=256, kernel_size=3),
            
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            
            ResidualBlock(channels=512, kernel_size=3),
            
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Flatten()
        )
        
        self.head = nn.Sequential(
            nn.Linear(in_features=64 * 4 * 4, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=num_class)
        )
        
        
    def forward(self, x):
        
        return self.head(self.model(x))
    
class SmallResnetDepthwiseReduce(nn.Module):
    
    def __init__(self, in_channels : int = 3, num_class : int = 8):
        super().__init__()
        
        self.model = nn.Sequential(
            
            DepthwiseSeparableConv(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            ResidualBlockDepthwise(channels=64, kernel_size=3),
            
            DepthwiseSeparableConv(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            
            ResidualBlockDepthwise(channels=128, kernel_size=3),
            
            DepthwiseSeparableConv(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            
            ResidualBlockDepthwise(channels=256, kernel_size=3),
            
            DepthwiseSeparableConv(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            
            ResidualBlockDepthwise(channels=512, kernel_size=3),
            
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Flatten()
        )
        
        self.head = nn.Sequential(
            nn.Linear(in_features=64 * 4 * 4, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=num_class)
        )
        
        
    def forward(self, x):
        
        return self.head(self.model(x))    

class SmallResnetExtended(nn.Module):
    
    def __init__(self, in_channels : int = 3, num_class : int = 8):
        super().__init__()
        
        self.model = nn.Sequential(
            
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            ResidualBlock(channels=64, kernel_size=3),
            ResidualBlock(channels=64, kernel_size=3),
            ResidualBlock(channels=64, kernel_size=3),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            
            ResidualBlock(channels=128, kernel_size=3),
            ResidualBlock(channels=128, kernel_size=3),
            ResidualBlock(channels=128, kernel_size=3),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Flatten()
        )
        
        self.head = nn.Sequential(
            nn.Linear(in_features=32 * 4 * 4, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=num_class)
        )
        
        
    def forward(self, x):
        
        return self.head(self.model(x))  
    
class SmallResnetDepthwiseExtended(nn.Module):
    
    def __init__(self, in_channels : int = 3, num_class : int = 8):
        super().__init__()
        
        self.model = nn.Sequential(
            
            DepthwiseSeparableConv(in_channels=in_channels, out_channels=64, kernel_size=7, padding=3, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            ResidualBlockDepthwise(channels=64, kernel_size=3),
            ResidualBlockDepthwise(channels=64, kernel_size=3),
            ResidualBlockDepthwise(channels=64, kernel_size=3),
            
            DepthwiseSeparableConv(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            
            ResidualBlockDepthwise(channels=128, kernel_size=3),
            ResidualBlockDepthwise(channels=128, kernel_size=3),
            ResidualBlockDepthwise(channels=128, kernel_size=3),
            
            DepthwiseSeparableConv(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            
            DepthwiseSeparableConv(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Flatten()
        )
        
        self.head = nn.Sequential(
            nn.Linear(in_features=32 * 4 * 4, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=num_class)
        )
        
        
    def forward(self, x):
        
        return self.head(self.model(x))  
    
if __name__ == "__main__":
    model = SmallResnetDepthwiseExtended()
    print(summary(model, input_size=(1, 3, 224, 224), depth=2))
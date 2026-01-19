import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,stride=stride,padding=padding,groups=in_channels,bias=bias), #Depthwise
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=bias), #Pointwise
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.layer(x)
    
    
class ResidualBlock(nn.Module):
    
    def __init__(self, channels : int, kernel_size : int):
        super().__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(num_features=channels)
        )
        
        
    def forward(self, x):
        return F.relu(self.layer(x) + x)
    
    
class ResidualBlockDepthwise(nn.Module):
    
    def __init__(self, channels : int, kernel_size : int):
        super().__init__()
        
        self.layer = nn.Sequential(
            DepthwiseSeparableConv(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            DepthwiseSeparableConv(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding='same'),
        )
        
        
    def forward(self, x):
        return F.relu(self.layer(x) + x)

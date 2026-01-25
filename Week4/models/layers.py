import torch
import torch.nn as nn
import torch.nn.functional as F
from models.squeeze_excitation import SqueezeExcitation

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

class ResidualBlockSE(nn.Module):
    
    def __init__(self, channels : int, kernel_size : int):
        super().__init__()
        
        self.layer = nn.Sequential(
            DepthwiseSeparableConv(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding='same'),
            SqueezeExcitation(channels=channels, reduction=8),
            nn.ReLU(),
            DepthwiseSeparableConv(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding='same'),
            SqueezeExcitation(channels=channels, reduction=8),
        )
        
        
    def forward(self, x):
        return F.relu(self.layer(x) + x)

class ChannelAttention(nn.Module): #Paper CNNtention 4.3.3
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module): #Paper CNNtention 4.3.4
    def __init__(self, kernel_size=7): #kernel size of 7 is used in the paper, 3 is also accepted according to the authors
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module): #Apply channel and spatial attention 
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        scale_c = self.channel_att(x)
        x = x * scale_c
        
        scale_s = self.spatial_att(x)
        x = x * scale_s
        
        return x
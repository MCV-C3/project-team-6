import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,stride=stride,padding=padding,groups=in_channels,bias=bias), #Depthwise
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=bias), #Pointwise
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)

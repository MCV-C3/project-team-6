import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution: groups=in_channels
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels, 
            bias=bias
        )
        # Pointwise convolution: 1x1 convolution to combine channels
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=bias
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

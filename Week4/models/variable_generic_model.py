import torch
import torch.nn as nn
from models.layers import DepthwiseSeparableConv, ResidualBlockDepthwise, SqueezeExcitation, CBAM

class VariableGenericSE(nn.Module):

    def __init__(self, 
                 channels_list=[16, 24, 24, 24, 32, 32, 32, 32], #best layer config generic se
                 num_class=8, 
                 use_residuals=False, 
                 use_cbam=False, 
                 dropout_rate=0.0, 
                 dropout_start_layer=100): # 100 equals nothing
        super().__init__()
        
        layers = []
        max_pool_idx = [1, 3, 5] 
        
        in_channels = 3
        layers.append(DepthwiseSeparableConv(in_channels=in_channels, out_channels=channels_list[0], kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU())
        
        in_channels = channels_list[0]
        
        for idx, channel in enumerate(channels_list[1:]):
            
            # try residuals
            if use_residuals and in_channels == channel:
                layers.append(ResidualBlockDepthwise(channels=channel, kernel_size=3))
            else:
                layers.append(DepthwiseSeparableConv(in_channels=in_channels, out_channels=channel, kernel_size=3, padding='same'))
                layers.append(nn.ReLU())

            # try changing se for attention
            if idx+1 > 1 and idx+1 < len(channels_list)-1:
                if use_cbam:
                    layers.append(CBAM(in_channels=channel, reduction_ratio=8))
                else:
                    layers.append(SqueezeExcitation(channels=channel, reduction=8))
            
            # dropout
            if idx >= dropout_start_layer and idx+1 < len(channels_list)-1:
                layers.append(nn.Dropout2d(p=dropout_rate))
                
            if idx+1 in max_pool_idx:
                layers.append(nn.MaxPool2d(2))
                
            in_channels = channel
                
        layers.append(nn.AdaptiveAvgPool2d(1))
        
        self.backbone = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_channels, num_class)

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(1)
        return self.classifier(x)
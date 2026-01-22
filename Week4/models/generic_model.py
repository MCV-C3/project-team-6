import torch
import torch.nn as nn
from torchinfo import summary
from models.layers import DepthwiseSeparableConv
from models.squeeze_excitation import SqueezeExcitation


class GenericSE(nn.Module):

    def __init__(self, in_channels: int = 3, channels : list = [8, 16, 16, 24], num_class: int = 8):
        super().__init__()
        
        layers = []
        max_pool_idx = [i for i in range(1, min(len(channels)-1, 6), 2)]
        
        layers.append(DepthwiseSeparableConv(in_channels=in_channels, out_channels=channels[0], kernel_size=3, stride=2, padding=1))
        layers.append(nn.ReLU())
        
        in_channels = channels[0]
        
        for idx, channel in enumerate(channels[1:]):
            
            layers.append(DepthwiseSeparableConv(in_channels=in_channels, out_channels=channel, kernel_size=3, padding='same'))
                
            if idx+1 > 1 and idx+1 < len(channels)-1:
                layers.append(SqueezeExcitation(channels=channel, reduction=8))
            
            layers.append(nn.ReLU())
            
            if idx > 5 and idx+1 < len(channels)-1:
                layers.append(nn.Dropout2d(p=0.1))
                
            if idx+1 in max_pool_idx:
                layers.append(nn.MaxPool2d(2))
                
            in_channels = channel
                
        layers.append(nn.AdaptiveAvgPool2d(1))
        
        self.backbone = nn.Sequential(*layers)
        self.classifier = self.head = nn.Linear(in_channels, num_class)


    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(1)
        return self.classifier(x)

if __name__ == "__main__":
    model = GenericSE()
    print(summary(model, input_size=(1, 3, 224, 224), depth=2))
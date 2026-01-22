import torch
import torch.nn as nn
from models.layers import DepthwiseSeparableConv
from torchinfo import summary

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            DepthwiseSeparableConv(in_channels, growth_rate, kernel_size=3, padding=1, bias=False) 
        )

    def forward(self, x):
        new_features = self.layer(x)
        return torch.cat([x, new_features], 1)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.layer(x)

class MicroDenseNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=8, growth_rate=4, block_layers=4):
        super(MicroDenseNet, self).__init__()
        
        # initial conv
        self.conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False)
        
        # dense block 1
        self.block1 = self._make_dense_block(inner_channels, growth_rate, layers=block_layers)
        inner_channels += growth_rate * block_layers # Los canales aumentan
        
        # transition
        out_channels = inner_channels // 2
        self.trans1 = TransitionLayer(inner_channels, out_channels)
        inner_channels = out_channels
        
        # dense block 2
        self.block2 = self._make_dense_block(inner_channels, growth_rate, layers=block_layers)
        inner_channels += growth_rate * block_layers
        
        # transition
        out_channels = inner_channels // 2
        self.trans2 = TransitionLayer(inner_channels, out_channels)
        inner_channels = out_channels
        
        # dense block 3
        self.block3 = self._make_dense_block(inner_channels, growth_rate, layers=block_layers)
        inner_channels += growth_rate * block_layers
        
        # classifier
        self.bn_final = nn.BatchNorm2d(inner_channels)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(inner_channels, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_dense_block(self, in_channels, growth_rate, layers):
        block = []
        for i in range(layers):
            block.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        return nn.Sequential(*block)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.trans1(x)
        x = self.block2(x)
        x = self.trans2(x)
        x = self.block3(x)
        
        x = self.relu(self.bn_final(x))
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = MicroDenseNet(growth_rate=6, block_layers=3)
    print(summary(model, input_size=(1, 3, 224, 224)))
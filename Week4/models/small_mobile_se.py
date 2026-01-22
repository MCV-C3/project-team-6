import torch
import torch.nn as nn
from torchinfo import summary
from models.layers import DepthwiseSeparableConv
from models.squeeze_excitation import SqueezeExcitation


class TinyMobileSE2k(nn.Module):

    def __init__(self, in_channels: int = 3, num_class: int = 8):
        super().__init__()

        self.model = nn.Sequential(
            # Early downsampling
            DepthwiseSeparableConv(in_channels, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            DepthwiseSeparableConv(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Refinement stage
            DepthwiseSeparableConv(16, 16, kernel_size=3, stride=1, padding=1),
            SqueezeExcitation(16, reduction=8),
            nn.ReLU(),

            # Final expansion
            DepthwiseSeparableConv(16, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.head = nn.Linear(24, num_class)

    def forward(self, x):
        x = self.model(x)
        x = x.flatten(1)
        return self.head(x)

class TinyMobileSEStartBig(nn.Module):

    def __init__(self, in_channels: int = 3, num_class: int = 8):
        super().__init__()

        self.model = nn.Sequential(
            # Keep full resolution initially
            DepthwiseSeparableConv(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            DepthwiseSeparableConv(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            DepthwiseSeparableConv(16, 16, kernel_size=3, stride=1, padding=1),
            SqueezeExcitation(16, reduction=8),
            nn.ReLU(),

            DepthwiseSeparableConv(16, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.head = nn.Linear(24, num_class)

    def forward(self, x):
        x = self.model(x)
        x = x.flatten(1)
        return self.head(x)

class TinyMobileSEExtended(nn.Module):

    def __init__(self, in_channels: int = 3, num_class: int = 8):
        super().__init__()

        self.model = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            DepthwiseSeparableConv(8, 16, kernel_size=3, stride=2, padding=1),
            SqueezeExcitation(16, reduction=8),
            nn.ReLU(),

            DepthwiseSeparableConv(16, 16, kernel_size=3, stride=1, padding=1),
            SqueezeExcitation(16, reduction=8),
            nn.ReLU(),

            DepthwiseSeparableConv(16, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.head = nn.Linear(24, num_class)

    def forward(self, x):
        x = self.model(x)
        x = x.flatten(1)
        return self.head(x)

class TinyMobileSE2kExpansion(nn.Module):

    def __init__(self, in_channels: int = 3, num_class: int = 8):
        super().__init__()

        self.model = nn.Sequential(
            # Early downsampling
            DepthwiseSeparableConv(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Refinement stage
            DepthwiseSeparableConv(16, 32, kernel_size=3, stride=1, padding=1),
            SqueezeExcitation(32, reduction=8),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Final expansion
            DepthwiseSeparableConv(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1)
        )

        self.head = nn.Linear(16, num_class)

    def forward(self, x):
        x = self.model(x)
        x = x.flatten(1)
        return self.head(x)

if __name__ == "__main__":
    model = TinyMobileSE2kExpansion()
    print(summary(model, input_size=(1, 3, 224, 224), depth=2))
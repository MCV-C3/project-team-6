import torch
from torch import nn

# TODO: Maybe try this? https://kornia.readthedocs.io/en/latest/augmentation.module.html
class AugmentationOnGPU(nn.Module):
    def __init__(self):
        super().__init__()

    def random_horizontal_flip(self, x):
        batch_size = x.size(0)
        flip_mask = torch.rand(batch_size, device=x.device) < 0.5
        x[flip_mask] = torch.flip(x[flip_mask], dims=[3])
        return x

    def random_color_jitter(self, x):
        brightness = 0.2
        contrast = 0.2

        B, C, H, W = x.shape
        if brightness > 0:
            b = (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * (2 * brightness)
            x = x + b

        if contrast > 0:
            means = x.mean(dim=(2, 3), keepdim=True)
            c = 1.0 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * (2 * contrast)
            x = (x - means) * c + means

        x = torch.clamp(x, 0, 1)
        return x

    def forward(self, x):
        x = self.random_horizontal_flip(x)
        x = self.random_color_jitter(x)
        return x

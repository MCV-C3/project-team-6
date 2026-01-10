import torch
from torch import nn
import kornia.augmentation as ka

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


full_augmentation = nn.Sequential(
    ka.RandomGaussianBlur(kernel_size=(7, 7), sigma=(0.5, 0.5), p=0.1),
    ka.RandomRotation(degrees=(-10, 10)),
    ka.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0), ratio=(1.0, 1.0)),
    ka.ColorJiggle(0.2, 0.2, 0.2, 0.2),
    ka.RandomHorizontalFlip(),
    ka.RandomGrayscale(),
    # # ka.Resize(size=FINAL_SIZE)
)

def make_full_augmentation(final_size: tuple[int, int]):
    return ka.AugmentationSequential(
        ka.RandomHorizontalFlip(),
        ka.RandomRotation(degrees=(-10, 10)),
        ka.RandomAffine(degrees=0, shear=cfg.shear,
                            translate=(cfg.translate_x, cfg.translate_y),
                            scale=(cfg.scale, cfg.scale)),
        ka.RandomResizedCrop(size=final_size, scale=(0.5, 1.0), ratio=(1.0, 1.0)),
        ka.RandomGaussianBlur(kernel_size=(7, 7), sigma=(0.5, 0.5), p=0.1),
        ka.ColorJiggle(0.2, 0.2, 0.2, 0.2, p=0.5),
        ka.RandomGrayscale(),
        ka.Resize(size=final_size)
    )

import torch
from torch import nn
import torch.nn.functional as F

from models.descriptor_classifier import DescriptorClassifier

class PatchDescriptionBlock(nn.Module):
    def __init__(self, *, patch_size: tuple[int, int], in_channels: int, out_channels: int):
        super(PatchDescriptionBlock, self).__init__()
        
        self.description = nn.Sequential(
            nn.Linear(in_features=in_channels * patch_size[0] * patch_size[1], out_features=out_channels),
            nn.GELU(),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.GELU(),
        )
        
        self.patch_size = patch_size
        
    # psize 16x16 im 256
    # [B, 3, 256, 256] -> [Bx16x16, 3, 16, 16] -> [B, 128, 16, 16]
    
    def extract_patches(self, x: torch.Tensor):
        B, C, H, W = x.shape
        ps = self.patch_size
        
        assert H % ps[0] == 0 and W % ps[1] == 0, f"Image size ({H}, {W}) must be divisible by patch size ({ps[0]}, {ps[1]})."
        
        patches = F.unfold(x, kernel_size=ps, stride=ps)

        patches = patches.transpose(1, 2)
        return patches
    
    def forward(self, x: torch.Tensor):
        batch_size, C, H, W = x.shape

        patches = self.extract_patches(x)
        num_patches = patches.shape[1]

        patches_flat = patches.reshape(-1, patches.shape[2])
        descriptors_flat = self.description(patches_flat)
        descriptor_dim = descriptors_flat.shape[1]
        descriptors = descriptors_flat.reshape(batch_size, descriptor_dim, H // self.patch_size[0], W // self.patch_size[1])
        return descriptors
    

def make_pyramidal_default() -> DescriptorClassifier:
    description = nn.Sequential(
        PatchDescriptionBlock(patch_size=(16, 16), in_channels=3, out_channels=128),
        PatchDescriptionBlock(patch_size=(7, 7), in_channels=128, out_channels=1024),
        PatchDescriptionBlock(patch_size=(2, 2), in_channels=1024, out_channels=4096),
        nn.Flatten()
    )
    
    classification = nn.Sequential(
        nn.Linear(in_features=4096, out_features=4096),
        nn.GELU(),
        nn.Linear(in_features=4096, out_features=11),
    )
    
    return DescriptorClassifier(
        description, classification,
    )

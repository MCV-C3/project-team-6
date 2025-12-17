import torch
from torch import nn
import torch.nn.functional as F

from models.descriptor_classifier import DescriptorClassifier

class PatchDescriptionBlock(nn.Module):
    def __init__(self, *, patch_size: tuple[int, int], in_channels: int, out_channels: int, dropout: float = 0.0):
        super(PatchDescriptionBlock, self).__init__()

        layers = [
            nn.Linear(in_features=in_channels * patch_size[0] * patch_size[1], out_features=out_channels),
            nn.GELU(),
        ]
        
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        layers.extend([
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.GELU(),
        ])

        self.description = nn.Sequential(*layers)
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


class DoubleDescriptorClassifier(nn.Module):
    def __init__(self, patch_description_head: nn.Module, image_descriptor_head: nn.Module, classification_head: nn.Module):
        super(DoubleDescriptorClassifier, self).__init__()

        self.patch_description_head = patch_description_head
        self.image_descriptor_head = image_descriptor_head
        self.classification_head = classification_head

    def forward(self, x):
        patch_descriptors = self.patch_description_head(x)
        image_descriptor = self.image_descriptor_head(patch_descriptors)
        output = self.classification_head(image_descriptor)
        return output

    def get_patch_descriptors(self, x):
        return self.patch_description_head(x)

    def get_image_descriptor(self, x):
        patch_descriptors = self.patch_description_head(x)
        return self.image_descriptor_head(patch_descriptors)

def make_double_descriptor_custom(
    first_patch_size: tuple[int, int],
    first_out_channels: int,
    pyramid_configs: list[tuple[tuple[int, int], int]],
    final_descriptor_dim: int,
    num_classes: int = 11,
    dropout: float = 0.0
) -> DoubleDescriptorClassifier:
    patch_description_head = PatchDescriptionBlock(
        patch_size=first_patch_size,
        in_channels=3,
        out_channels=first_out_channels,
        dropout=dropout
    )

    image_descriptor_modules = []
    in_channels = first_out_channels
    for patch_size, out_channels in pyramid_configs:
        image_descriptor_modules.append(
            PatchDescriptionBlock(
                patch_size=patch_size,
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout
            )
        )
        in_channels = out_channels
    image_descriptor_modules.append(nn.Flatten())
    image_descriptor_head = nn.Sequential(*image_descriptor_modules)

    classification_layers = [
        nn.Linear(in_features=in_channels, out_features=final_descriptor_dim),
        nn.GELU(),
    ]
    if dropout > 0:
        classification_layers.append(nn.Dropout(p=dropout))
    classification_layers.append(nn.Linear(in_features=final_descriptor_dim, out_features=num_classes))
    classification_head = nn.Sequential(*classification_layers)

    return DoubleDescriptorClassifier(
        patch_description_head,
        image_descriptor_head,
        classification_head
    )

def make_double_descriptor_prototype(dropout: float = 0.0, num_classes: int = 11) -> DoubleDescriptorClassifier:
    patch_description_head = PatchDescriptionBlock(
        patch_size=(16, 16),
        in_channels=3,
        out_channels=128,
        dropout=dropout
    )

    image_descriptor_modules = [
        PatchDescriptionBlock(patch_size=(7, 7), in_channels=128, out_channels=512, dropout=dropout),
        PatchDescriptionBlock(patch_size=(2, 2), in_channels=512, out_channels=1024, dropout=dropout),
        nn.Flatten(),
        nn.BatchNorm1d(1024),
    ]
    image_descriptor_head = nn.Sequential(*image_descriptor_modules)

    classification_layers = [
        nn.Linear(in_features=1024, out_features=num_classes)
    ]
    classification_head = nn.Sequential(*classification_layers)

    return DoubleDescriptorClassifier(
        patch_description_head,
        image_descriptor_head,
        classification_head
    )

def make_double_descriptor_default(dropout: float = 0.0, num_classes: int = 11) -> DoubleDescriptorClassifier:
    patch_description_head = PatchDescriptionBlock(
        patch_size=(16, 16),
        in_channels=3,
        out_channels=128,
        dropout=dropout
    )

    image_descriptor_modules = [
        PatchDescriptionBlock(patch_size=(7, 7), in_channels=128, out_channels=1024, dropout=dropout),
        PatchDescriptionBlock(patch_size=(2, 2), in_channels=1024, out_channels=4096, dropout=dropout),
        nn.Flatten()
    ]
    image_descriptor_head = nn.Sequential(*image_descriptor_modules)

    classification_layers = [
        nn.Linear(in_features=4096, out_features=4096),
        nn.GELU(),
    ]
    if dropout > 0:
        classification_layers.append(nn.Dropout(p=dropout))
    classification_layers.append(nn.Linear(in_features=4096, out_features=num_classes))
    classification_head = nn.Sequential(*classification_layers)

    return DoubleDescriptorClassifier(
        patch_description_head,
        image_descriptor_head,
        classification_head
    )

def make_pyramidal_default(dropout: float = 0.0) -> DescriptorClassifier:
    description = nn.Sequential(
        PatchDescriptionBlock(patch_size=(16, 16), in_channels=3, out_channels=128, dropout=dropout),
        PatchDescriptionBlock(patch_size=(7, 7), in_channels=128, out_channels=1024, dropout=dropout),
        PatchDescriptionBlock(patch_size=(2, 2), in_channels=1024, out_channels=4096, dropout=dropout),
        nn.Flatten()
    )

    classification_layers = [
        nn.Linear(in_features=4096, out_features=4096),
        nn.GELU(),
    ]
    if dropout > 0:
        classification_layers.append(nn.Dropout(p=dropout))
    classification_layers.append(nn.Linear(in_features=4096, out_features=11))

    classification = nn.Sequential(*classification_layers)

    return DescriptorClassifier(
        description, classification,
    )


def make_pyramidal_fine_to_coarse(dropout: float = 0.0) -> DescriptorClassifier:
    description = nn.Sequential(
        PatchDescriptionBlock(patch_size=(2, 2), in_channels=3, out_channels=64, dropout=dropout),
        PatchDescriptionBlock(patch_size=(4, 4), in_channels=64, out_channels=256, dropout=dropout),
        PatchDescriptionBlock(patch_size=(7, 7), in_channels=256, out_channels=1024, dropout=dropout),
        PatchDescriptionBlock(patch_size=(4, 4), in_channels=1024, out_channels=4096, dropout=dropout),
        nn.Flatten()
    )

    classification_layers = [
        nn.Linear(in_features=4096, out_features=4096),
        nn.GELU(),
    ]
    if dropout > 0:
        classification_layers.append(nn.Dropout(p=dropout))
    classification_layers.append(nn.Linear(in_features=4096, out_features=11))

    classification = nn.Sequential(*classification_layers)

    return DescriptorClassifier(
        description, classification,
    )

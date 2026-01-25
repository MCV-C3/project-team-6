import torch
import torch.nn as nn
from typing import Literal


class PatchBasedClassifier(nn.Module):
    def __init__(
        self,
        descriptor_head: nn.Module,
        classification_head: nn.Module,
        patch_size: int,
        stride: int = None,
        merge_strategy: Literal["mean", "max", "median", "voting"] = "mean",
        num_classes: int = 11,
    ):
        super(PatchBasedClassifier, self).__init__()

        self.descriptor_head = descriptor_head
        self.classification_head = classification_head
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.merge_strategy = merge_strategy
        self.num_classes = num_classes

    def _extract_patches(self, x):
        B, C, H, W = x.shape

        # Calculate number of patches
        num_patches_h = (H - self.patch_size) // self.stride + 1
        num_patches_w = (W - self.patch_size) // self.stride + 1

        # Validate that patches fit exactly if stride equals patch_size
        if self.stride == self.patch_size:
            assert H % self.patch_size == 0, \
                f"Image height {H} is not divisible by patch size {self.patch_size}"
            assert W % self.patch_size == 0, \
                f"Image width {W} is not divisible by patch size {self.patch_size}"
        else:
            # For overlapping patches, check that we can extract at least one patch
            assert H >= self.patch_size, \
                f"Image height {H} is smaller than patch size {self.patch_size}"
            assert W >= self.patch_size, \
                f"Image width {W} is smaller than patch size {self.patch_size}"

            # Check that the last patch fits exactly
            assert (H - self.patch_size) % self.stride == 0, \
                f"Image height {H} with patch size {self.patch_size} and stride {self.stride} " \
                f"does not allow exact tiling. Last patch would be incomplete."
            assert (W - self.patch_size) % self.stride == 0, \
                f"Image width {W} with patch size {self.patch_size} and stride {self.stride} " \
                f"does not allow exact tiling. Last patch would be incomplete."

        patches = x.unfold(2, self.patch_size, self.stride)
        patches = patches.unfold(3, self.patch_size, self.stride)

        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()

        num_patches = num_patches_h * num_patches_w
        patches = patches.view(B * num_patches, C, self.patch_size, self.patch_size)

        return patches, num_patches

    def _merge_predictions(self, logits, num_patches, descriptors=None):
        B = logits.shape[0] // num_patches

        logits = logits.view(B, num_patches, self.num_classes)

        if self.merge_strategy == "mean":
            return logits.mean(dim=1)

        elif self.merge_strategy == "max":
            return logits.max(dim=1)[0]

        elif self.merge_strategy == "median":
            return logits.median(dim=1)[0]

        elif self.merge_strategy == "voting":
            # Soft voting: average the softmax probabilities across patches
            # This is differentiable and can be used during training
            probabilities = torch.softmax(logits, dim=2)
            averaged_probs = probabilities.mean(dim=1)
            # Convert back to logits for loss computation
            # Add small epsilon to avoid log(0)
            return torch.log(averaged_probs + 1e-10)

        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")

    def forward(self, x):
        patches, num_patches = self._extract_patches(x)

        descriptors = self.descriptor_head(patches)

        logits = self.classification_head(descriptors)

        merged_logits = self._merge_predictions(logits, num_patches, descriptors)

        return merged_logits

    def get_descriptors(self, x):
        patches, num_patches = self._extract_patches(x)

        descriptors = self.descriptor_head(patches)

        return descriptors


def make_patch_model(
    input_channels: int = 3,
    patch_size: int = 32,
    stride: int = None,
    descriptor_widths: list[int] = [256, 128],
    num_classes: int = 11,
    merge_strategy: Literal["mean", "max", "median", "voting"] = "mean",
) -> PatchBasedClassifier:
    """
    Convenience function to create a patch-based classifier with MLP heads.

    Args:
        input_channels: Number of input channels (e.g., 3 for RGB)
        patch_size: Size of square patches
        stride: Stride for patch extraction. If None, uses patch_size (non-overlapping)
        descriptor_widths: List of hidden layer widths for descriptor MLP
        num_classes: Number of output classes
        merge_strategy: Strategy to merge patch predictions

    Returns:
        PatchBasedClassifier instance

    Example:
        >>> model = make_patch_model(
        ...     input_channels=3,
        ...     patch_size=32,
        ...     descriptor_widths=[512, 256],
        ...     num_classes=11,
        ...     merge_strategy="mean"
        ... )
    """
    # Calculate input dimension for descriptor head
    input_dim = input_channels * patch_size * patch_size

    # Build descriptor head (MLP)
    descriptor_modules = [nn.Flatten()]
    prev_dim = input_dim
    for width in descriptor_widths:
        descriptor_modules.append(nn.Linear(prev_dim, width))
        descriptor_modules.append(nn.ReLU())
        prev_dim = width
    descriptor_head = nn.Sequential(*descriptor_modules)

    # Build classification head (single linear layer)
    classification_head = nn.Linear(prev_dim, num_classes)

    return PatchBasedClassifier(
        descriptor_head=descriptor_head,
        classification_head=classification_head,
        patch_size=patch_size,
        stride=stride,
        num_classes=num_classes,
        merge_strategy=merge_strategy,
    )

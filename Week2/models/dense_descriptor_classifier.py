from torch import nn
import torch.nn.functional as F

class DenseDescriptorClassifier(nn.Module):
    def __init__(self, *, patch_size: tuple[int, int], descriptor_head: nn.Module, classification_head: nn.Module):
        super(DenseDescriptorClassifier, self).__init__()

        self.patch_size = patch_size
        self.descriptor_head = descriptor_head
        self.classification_head = classification_head

    def extract_patches(self, x):
        B, C, H, W = x.shape
        ps = self.patch_size
        
        assert H % ps[0] == 0 and W % ps[1] == 0, f"Image size ({H}, {W}) must be divisible by patch size ({ps[0]}, {ps[1]})."
        
        patches = F.unfold(x, kernel_size=ps, stride=ps)

        patches = patches.transpose(1, 2)
        return patches
        
    def forward(self, x):
        batch_size = x.shape[0]

        patches = self.extract_patches(x)
        num_patches = patches.shape[1]

        patches_flat = patches.reshape(-1, patches.shape[2])
        descriptors_flat = self.descriptor_head(patches_flat)
        descriptor_dim = descriptors_flat.shape[1]
        descriptors = descriptors_flat.reshape(batch_size, num_patches, descriptor_dim)
        
        aggregated = descriptors.reshape(batch_size, num_patches * descriptor_dim)
        
        output = self.classification_head(aggregated)
        
        return output

    def get_descriptors(self, x):
        pass # TODO
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2  as F
from typing import Optional, Tuple


def get_loaders(image_size : Optional[Tuple[int]]):
    
    if not image_size:
        image_size = (224, 224)

    torch.manual_seed(42)

    train_transformation = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=image_size),
    ])
    test_transformation = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=image_size),
    ])

    data_train = ImageFolder("../data/places_reduced/train", transform=train_transformation)
    data_test = ImageFolder("../data/places_reduced/val", transform=test_transformation)

    train_loader = DataLoader(data_train, batch_size=256, pin_memory=True, shuffle=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
    test_loader = DataLoader(data_test, batch_size=128, pin_memory=True, shuffle=False, num_workers=8, prefetch_factor=4, persistent_workers=True)
    
    return train_loader, test_loader
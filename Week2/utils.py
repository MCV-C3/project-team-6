import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2  as F
from typing import Optional, Tuple
import argparse


def get_loaders(image_size : Optional[Tuple[int, int]] = None, train_batch_size: int = 256, resize_train: bool = True, resize_test: bool = True):

    torch.manual_seed(42)

    # I do this to skip the image
    train_transforms = [
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
    ]
    test_transforms = [
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
    ]

    # Vaya lio esto, pero es para poder hacer resize solo de test si hacemos augmentation en train
    if image_size is not None and resize_train:
        train_transforms.append(F.Resize(size=image_size))
    if image_size is not None and resize_test:
        test_transforms.append(F.Resize(size=image_size))

    train_transformation = F.Compose(train_transforms)
    test_transformation = F.Compose(test_transforms)

    data_train = ImageFolder("../data/places_reduced/train", transform=train_transformation)
    data_test = ImageFolder("../data/places_reduced/val", transform=test_transformation)

    train_loader = DataLoader(data_train, batch_size=train_batch_size, pin_memory=True, shuffle=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
    test_loader = DataLoader(data_test, batch_size=128, pin_memory=True, shuffle=False, num_workers=8, prefetch_factor=4, persistent_workers=True)

    return train_loader, test_loader


def get_experiment_argument_parser():
    parser = argparse.ArgumentParser(description="Experiment parameters")

    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use (default: 0)')

    return parser


def set_device(gpu_id: int = 0):
    """
    Set the GPU device to use for torch operations.

    Args:
        gpu_id (int): GPU ID to use. If CUDA is not available, will fall back to CPU.

    Returns:
        torch.device: The device object to use for model and tensors.
    """
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    return device
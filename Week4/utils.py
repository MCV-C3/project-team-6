import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms.v2  as F
from typing import Optional, Tuple
import argparse

import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl

class Null:
    def __getattr__(self, name):
        return self

    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getitem__(self, key):
        return self

    def __setitem__(self, key, value):
        pass

    def __bool__(self):
        return False
    
    def __iter__(self):
        return iter([])

    def __repr__(self):
        return "<Null>"


class InMemoryDataset(Dataset):
    def __init__(self, source: Dataset, device, transform=None):
        self.images = [] 
        self.targets = []
        self.transform = transform
        self.device = device
        
        imgs = []
        targets = []

        for img, target in source:
            t = F.functional.pil_to_tensor(img)
            imgs.append(t)
            targets.append(target)

        self.images = torch.stack(imgs).to(device)
        self.targets = torch.tensor(targets).to(device)
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.images[idx]), self.targets[idx]
        return self.images[idx], self.targets[idx]


def get_loaders(image_size : Optional[Tuple[int, int]] = None, 
                train_batch_size: int = 256, 
                resize_train: bool = False, 
                resize_test: bool = True, 
                train_folder: str = "~/mcv/datasets/C3/2425/MIT_small_train_1/train", 
                test_folder: str = "~/mcv/datasets/C3/2425/MIT_small_train_1/test"):

    # I do this to skip the image
    train_transforms = [
        # F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
    ]
    test_transforms = [
        # F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
    ]

    # Vaya lio esto, pero es para poder hacer resize solo de test si hacemos augmentation en train
    if image_size is not None and resize_train:
        train_transforms.append(F.Resize(size=image_size))
    if image_size is not None and resize_test:
        test_transforms.append(F.Resize(size=image_size))

    train_transformation = F.Compose(train_transforms)
    test_transformation = F.Compose(test_transforms)
    
    # This dataset is very small, we can store it fully in VRAM!!!!!
    data_train = InMemoryDataset(ImageFolder(train_folder), "cuda", transform=train_transformation)
    data_test = InMemoryDataset(ImageFolder(test_folder), "cuda", transform=test_transformation)

    train_loader = DataLoader(data_train, batch_size=train_batch_size, pin_memory=False, shuffle=True, num_workers=0, prefetch_factor=None, persistent_workers=False)
    test_loader = DataLoader(data_test, batch_size=128, pin_memory=False, shuffle=False, num_workers=0, prefetch_factor=None, persistent_workers=False)

    # THE OLD WAY JUST IN CASE
    # data_train = ImageFolder("../data/places_reduced/train", transform=train_transformation)
    # data_test = ImageFolder("../data/places_reduced/val", transform=test_transformation)

    # train_loader = DataLoader(data_train, batch_size=train_batch_size, pin_memory=True, shuffle=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
    # test_loader = DataLoader(data_test, batch_size=128, pin_memory=True, shuffle=False, num_workers=8, prefetch_factor=4, persistent_workers=True)
        

    return train_loader, test_loader


def get_experiment_argument_parser():
    parser = argparse.ArgumentParser(description="Experiment parameters")

    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--dry', default=False, action='store_true', help="If present, do not log to wandb nor save or generate any file.")
    parser.add_argument('--augmentation', default=False, action='store_true', help="If present, does data augmentation on the training.")
    parser.add_argument('--train_folder', default="~/mcv/datasets/C3/2425/MIT_small_train_1/train", help="The folder that contains the training data.")
    parser.add_argument('--test_folder', default="~/mcv/datasets/C3/2425/MIT_small_train_1/test", help="The folder that contains the training data.")
    
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

def init_wandb_run(*args, **kwargs):
    dry = kwargs.pop("dry", False)
    if not dry:
        return wandb.init(*args, **kwargs)
    else:
        return Null()

def get_trainer(logger, patience : int = 15, min_delta : float = 0.001, epochs : int = 500):
    
    # early_stop_callback = EarlyStopping(
    #     monitor="test_loss",  # metric to monitor
    #     patience=patience,          # stop if no improvement for patience epochs
    #     min_delta=min_delta,
    #     verbose=True,
    #     mode="min"           # because lower test_loss is better
    # )

    checkpoint_callback = ModelCheckpoint(
        monitor="test_acc",          # metric to monitor
        mode="max",                  # save the lowest val_loss
        save_top_k=1,                # only save the best model
        dirpath="checkpoints/",      # folder to save
        filename="best_model"        # name of the checkpoint file
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1
    )
    
    return trainer
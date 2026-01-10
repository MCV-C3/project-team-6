import torch
import torch.nn as nn
from torchvision import models
import utils
from augmentation import make_full_augmentation
import wandb
from copy import deepcopy
import kornia.augmentation as ka
from experiments.squeeze_excitation_experiment import create_model_for_experiment

from models.base import WraperModel
from pipeline import experiment

argparser = utils.get_experiment_argument_parser()

args, _ = argparser.parse_known_args()

EPOCHS = args.epochs
IMG_SIZE = 224
dry = args.dry
device = utils.set_device(args.gpu_id)

run = utils.init_wandb_run(
            dry=dry,
            entity="mcv-team-6",
            project="C3-Week3",
            name="Definitive training",
            config={
                "architecture": f"DenseNet121-Squeeze-Excitation",
                "experiment_type": 'Definitive Parameters',
                "epochs": EPOCHS,
                "image_size": IMG_SIZE,
            }
        )

model = create_model_for_experiment("Sweep", num_classes=8)

LR = 2.1729679390519813e-05
batch_size = 32
weight_decay = 1.461506239963124e-05
label_smoothing = 0.02254139921835141
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)

augmentation = ka.AugmentationSequential(
        ka.RandomHorizontalFlip(p=0.04855768013650663),
        ka.RandomRotation(degrees=16,
                          p=0.13009412985538538),
        ka.RandomAffine(degrees=0, shear=29,
                            translate=(0.3584182906605954, 0.035543949656199385),
                            scale=(1, 1)),
        ka.RandomResizedCrop(size=(IMG_SIZE,IMG_SIZE), scale=(0.4382924444152586, 1.0), ratio=(1.0, 1.0), p=0.1939592095722116),
        ka.RandomGaussianBlur(kernel_size=(9, 9), sigma=(1.190437323963532, 1.190437323963532), p=0.442038986973481),
        ka.ColorJitter(brightness=1.1237698676931558,
                           contrast=0.7852337555662053,
                           saturation=1.20141373248246,
                           hue=0.06272478548259297,
                           p=0.2029549731865357),
        ka.RandomGrayscale(p=0.331441327268726),
        ka.Resize(size=(IMG_SIZE, IMG_SIZE))
    )

train_loader, test_loader = utils.get_loaders(
    image_size=(IMG_SIZE, IMG_SIZE),
    resize_train=True,
    resize_test=True,
    train_batch_size=batch_size,
    train_folder=args.train_folder,
    test_folder=args.test_folder,
)

loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        

experiment_name = "Definitive model"
experiment(
    experiment_name,
    model=model,
    optimizer=optimizer,
    criterion=loss,
    epochs=EPOCHS,
    train_loader=train_loader,
    test_loader=test_loader,
    augmentation=augmentation,
    wandb_run=run,
    device=device,
    early_stopping_patience=15,
    early_stopping_min_delta=0.001,
    )

run.finish()

del model
del optimizer
torch.cuda.empty_cache()
wandb.join()
del train_loader
del test_loader
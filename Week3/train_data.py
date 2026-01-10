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

args, unknown = argparser.parse_known_args()
sweep_config = {}
for arg in unknown:
    if arg.startswith("--"):
        key, val = arg[2:].split("=")
        try:
            # Convert numeric values automatically
            val = float(val)
            if val.is_integer():
                val = int(val)
        except ValueError:
            pass  # leave as string
        sweep_config[key] = val

EPOCHS = args.epochs
IMG_SIZE = 224
dry = args.dry
device = utils.set_device(args.gpu_id)

run = utils.init_wandb_run(
            dry=dry,
            entity="mcv-team-6",
            project="C3-Week3",
            name="Sweep run",
            config=sweep_config
        )

model = create_model_for_experiment("Sweep", num_classes=8)

cfg = wandb.config

LR = 0.00004037030818574088
batch_size = 32
weight_decay = 0.0003625524454634211
label_smoothing = 0.05754955518817498
optimizer = torch.optim.Adamax(model.parameters(), lr=LR, weight_decay=weight_decay)

augmentation = ka.AugmentationSequential(
        ka.RandomHorizontalFlip(p=cfg.flip),
        ka.RandomRotation(degrees=cfg.rotation, p=cfg.rotate),
        ka.RandomAffine(degrees=0, shear=cfg.shear,
                            translate=(cfg.translate_x, cfg.translate_y),
                            scale=(cfg.scale, cfg.scale)),
        ka.RandomResizedCrop(size=(IMG_SIZE,IMG_SIZE), scale=(cfg.crop_scale, 1.0), ratio=(1.0, 1.0), p=cfg.crop),
        ka.RandomGaussianBlur(kernel_size=(cfg.gaussian_blur_kernel, cfg.gaussian_blur_kernel), sigma=(cfg.gaussian_noise_std, cfg.gaussian_noise_std), p=cfg.blur),
        ka.ColorJitter(brightness=cfg.brightness,
                           contrast=cfg.contrast,
                           saturation=cfg.saturation,
                           hue=cfg.hue,
                           p=cfg.jiggle),
        ka.RandomGrayscale(p=cfg.grey),
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
        

experiment_name = "Sweep_data run"
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
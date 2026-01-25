import torch
import torch.nn as nn
from torchvision import models
import utils
from augmentation import make_full_augmentation
import wandb
from copy import deepcopy
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

cfg = wandb.config

LR = cfg.lr
batch_size = cfg.batch_size
weight_decay = cfg.weight_decay
label_smoothing = cfg.label_smoothing
optimizer_name = cfg.optimizer

augmentation = make_full_augmentation((IMG_SIZE, IMG_SIZE))

train_loader, test_loader = utils.get_loaders(
    image_size=(IMG_SIZE, IMG_SIZE),
    resize_train=True,
    resize_test=True,
    train_batch_size=batch_size,
    train_folder=args.train_folder,
    test_folder=args.test_folder,
)

model = create_model_for_experiment("Sweep", num_classes=8)

loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)

match optimizer_name:
    
    case 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=weight_decay)
        
    case 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=LR, weight_decay=weight_decay)
        
    case 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=LR, weight_decay=weight_decay)
        
    case 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    
    case 'Adamax':
        optimizer = torch.optim.Adamax(model.parameters(), lr=LR, weight_decay=weight_decay)
        
    case 'Nadam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=LR, weight_decay=weight_decay)
        
    case 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
        

experiment_name = "Sweep run"
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
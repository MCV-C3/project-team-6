import torch
from augmentation import make_full_augmentation
import utils
import wandb

from pipeline import experiment
from models.base_model import BaseModel


argparser = utils.get_experiment_argument_parser()
args = argparser.parse_args()

EPOCHS = args.epochs
SIDE = 32

dry = args.dry
device = utils.set_device(args.gpu_id)

DO_AUGMENTATION = True

train_loader, test_loader = utils.get_loaders(image_size=(SIDE, SIDE), resize_train=not DO_AUGMENTATION)

model = BaseModel([
    3*SIDE*SIDE,
    512,
    1024,
    512,
    11
])

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.0001)

augmentation = make_full_augmentation((SIDE, SIDE)) if DO_AUGMENTATION else None

run = utils.init_wandb_run(
    dry=dry,
    entity="mcv-team-6",
    project="C3-Week2",
    name=f"Imsize 16x16 with more complexity",
    config={
        "architecture": "BaseModel",
        "epochs": EPOCHS,
        "depth" : 3,
        "widths" : [512, 1024, 512]
    }
)

experiment(f"base_model_imsize_{SIDE}x{SIDE}_complexified",
    model=model,
    optimizer=optimizer,
    criterion=loss,
    epochs=EPOCHS,
    train_loader=train_loader,
    test_loader=test_loader,
    augmentation=augmentation,
    wandb_run=run,
    device=device,
)


run.finish()
wandb.join()

del train_loader
del test_loader
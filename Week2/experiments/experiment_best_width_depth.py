import torch
from augmentation import make_full_augmentation
import utils
import wandb

from models.descriptor_classifier import make_from_widths, make_like_simple
from pipeline import experiment
from models.base_model import BaseModel


argparser = utils.get_experiment_argument_parser()
args = argparser.parse_args()

EPOCHS = args.epochs
WIDTH = 512
DEPTH = 5
dry = args.dry
device = utils.set_device(args.gpu_id)

train_loader, test_loader = utils.get_loaders(image_size=(224, 224), resize_train=False)

model = make_from_widths(3*224*224, [512]*5, [11])
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.0001)


run = utils.init_wandb_run(
    dry=dry,
    entity="mcv-team-6",
    project="C3-Week2",
    name=f"Best model w={WIDTH} d={DEPTH}",
    config={
        "architecture": "BaseModel",
        "epochs": EPOCHS
    }
)

experiment(f"base_model_width_{WIDTH}_depth_{DEPTH}",
    model=model,
    optimizer=optimizer,
    criterion=loss,
    epochs=EPOCHS,
    train_loader=train_loader,
    test_loader=test_loader,
    augmentation=make_full_augmentation((224, 224)),
    wandb_run=run,
    device=device,
)


run.finish()
wandb.join()

del train_loader
del test_loader
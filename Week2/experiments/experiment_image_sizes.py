import torch
from augmentation import make_full_augmentation
import utils
import wandb

from models.descriptor_classifier import make_from_widths, make_like_simple
from pipeline import experiment
from models.base_model import BaseModel


argparser = utils.get_experiment_argument_parser()
argparser.add_argument('--depth', type=int, default=2, help='Depth of the hidden layers')
argparser.add_argument('--width', type=int, default=300, help='Width of the hidden layers')
argparser.add_argument('--imsize', type=int, default=224, help='Side length of the image')
args = argparser.parse_args()

EPOCHS = args.epochs
SIDE = args.imsize
WIDTH = args.width
DEPTH = args.depth

dry = args.dry
device = utils.set_device(args.gpu_id)

DO_AUGMENTATION = True

train_loader, test_loader = utils.get_loaders(image_size=(SIDE, SIDE), resize_train=not DO_AUGMENTATION)

model = make_from_widths(3*SIDE*SIDE, [WIDTH]*DEPTH, [11])
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.0001)

augmentation = make_full_augmentation((SIDE, SIDE)) if DO_AUGMENTATION else None

run = utils.init_wandb_run(
    dry=dry,
    entity="mcv-team-6",
    project="C3-Week2",
    name=f"(w={WIDTH} d={DEPTH}) Imsize variation : {SIDE}x{SIDE}",
    config={
        "architecture": "BaseModel",
        "epochs": EPOCHS
    }
)

experiment(f"base_model_imsize_{SIDE}x{SIDE}_width_{WIDTH}_depth_{DEPTH}",
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
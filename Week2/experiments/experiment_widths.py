import torch
import utils
import wandb

from models.descriptor_classifier import make_like_simple
from pipeline import experiment
from models.base_model import BaseModel


argparser = utils.get_experiment_argument_parser()
argparser.add_argument('--width', type=int, default=100, help='Width of the hidden layer')
args = argparser.parse_args()

EPOCHS = args.epochs
WIDTH = args.width

device = utils.set_device(args.gpu_id)

train_loader, test_loader = utils.get_loaders(image_size=(224, 224))
# model = BaseModel([
#     3*224*224,
#     WIDTH,
#     11])

model = make_like_simple(3*224*224, WIDTH, 11)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.0001)

run = wandb.init(
    entity="mcv-team-6",
    project="C3-Week2",
    name=f"Width only variation : {WIDTH}",
    config={
        "architecture": "BaseModel",
        "epochs": EPOCHS
    }
)

experiment(f"base_model_width_{WIDTH}",
    model=model,
    optimizer=optimizer,
    criterion=loss,
    epochs=EPOCHS,
    train_loader=train_loader,
    test_loader=test_loader,
    augmentation=None,
    wandb_run=run,
    device=device,
)


run.finish()
wandb.join()

del train_loader
del test_loader
import torch
import utils
import wandb

from models.descriptor_classifier import make_like_simple
from pipeline import experiment
from models.base_model import BaseModel
import time

# FIXME: esto crea un modelo de 2 capas solo, no sé si es que falta un 300
# model = BaseModel([
#     3 * 224 * 224,
#     300,
#     11
# ])

argparser = utils.get_experiment_argument_parser()
args = argparser.parse_args()

device = utils.set_device(args.gpu_id)

model = make_like_simple(input_d=3 * 224 * 224, hidden_d=300, output_d=11)
optimizer = torch.optim.Adam(model.parameters(), 0.0001)
loss = torch.nn.CrossEntropyLoss()

epochs = args.epochs

run = wandb.init(
    entity="mcv-team-6",
    project="C3-Week2",
    name="Baseline",
    config={
        "architecture": "BaseModel",
        "epochs": epochs
    }
)

train_loader, test_loader = utils.get_loaders(image_size=(224, 224))

experiment("test_run",
    model=model,
    optimizer=optimizer,
    criterion=loss,
    epochs=epochs,
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
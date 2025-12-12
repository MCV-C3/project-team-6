import torch
import utils
import wandb
from pipeline import experiment
from models.base_model import BaseModel


EPOCHS = 2
WIDTH = 512

train_loader, test_loader = utils.get_loaders(image_size=(224, 224))
model = BaseModel([
    3*224*224, 
    WIDTH, 
    11])

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

experiment("test_run",
    model=model,
    optimizer=optimizer,
    criterion=loss,
    epochs=EPOCHS,
    train_loader=train_loader,
    test_loader=test_loader,
    augmentation=None,
    wandb_run=run,
)


run.finish()
wandb.join()

del train_loader
del test_loader
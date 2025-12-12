import torch
import utils
import wandb
from pipeline import experiment
from models.base_model import BaseModel


EPOCHS = 2
DEPTH = 4
GPU_ID = 0

device = utils.set_device(GPU_ID)

train_loader, test_loader = utils.get_loaders(image_size=(224, 224))

widths = [3*224*224]

for i in range(DEPTH):
    widths.append(300)
    
widths.append(11)

model = BaseModel(widths=widths)

print(model)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.0001)

run = wandb.init(
    entity="mcv-team-6",
    project="C3-Week2",
    name=f"Depth only variation : {DEPTH}",
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
    device=device,
)

run.finish()
wandb.join()

del train_loader
del test_loader
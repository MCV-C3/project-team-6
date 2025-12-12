import torch
import wandb
from base import experiment
from models.descriptor_classifier import make_like_simple
import torchvision.transforms.v2  as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


model = make_like_simple(3 * 224 * 224, 300, 11)
optimizer = torch.optim.Adam(model.parameters(), 0.0001)
loss = torch.nn.CrossEntropyLoss()

run = wandb.init(
    entity="mcv-team-6",
    project="C3-Week2",
    name="Test Run",
    config={
        "architecture": "make_like_simple(3 * 224 * 224, 300, 11)",
        "epochs": 20
    }
)

FINAL_SIZE = (224, 224)
FINAL_C = 3

torch.manual_seed(42)

train_transformation = F.Compose([
    F.ToImage(),
    F.ToDtype(torch.float32, scale=True),
    F.Resize(size=FINAL_SIZE),
])
test_transformation = F.Compose([
    F.ToImage(),
    F.ToDtype(torch.float32, scale=True),
    F.Resize(size=FINAL_SIZE),
])

data_train = ImageFolder("../data/places_reduced/train", transform=train_transformation)
data_test = ImageFolder("../data/places_reduced/val", transform=test_transformation)

train_loader = DataLoader(data_train, batch_size=256, pin_memory=True, shuffle=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
test_loader = DataLoader(data_test, batch_size=128, pin_memory=True, shuffle=False, num_workers=8, prefetch_factor=4, persistent_workers=True)


experiment("test_run",
    model=model,
    optimizer=optimizer,
    criterion=loss,
    epochs=20,
    train_loader=train_loader,
    test_loader=test_loader,
    augmentation=None,
    wandb_run=run
)

run.finish()
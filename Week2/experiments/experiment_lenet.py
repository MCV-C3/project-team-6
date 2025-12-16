import torch
from augmentation import make_full_augmentation
import utils
import wandb

from models.patch_based_classifier import make_patch_model
from models.complex_models.lenet import LeNet
from pipeline import experiment


argparser = utils.get_experiment_argument_parser()
args = argparser.parse_args()

EPOCHS = args.epochs
IMG_SIZE = 224

dry = args.dry
device = utils.set_device(args.gpu_id)

train_loader, test_loader = utils.get_loaders(image_size=(IMG_SIZE, IMG_SIZE), resize_train=True)

# descriptor_widths = [WIDTH]
# for i in range(1, DEPTH):
#     descriptor_widths.append(max(128, WIDTH // (2 ** i)))

model = LeNet((IMG_SIZE, IMG_SIZE), 11)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.0001)

augmentation = make_full_augmentation((IMG_SIZE, IMG_SIZE))

run_name = f"LeNet"

run = utils.init_wandb_run(
    dry=dry,
    entity="mcv-team-6",
    project="C3-Week2",
    name=run_name,
    config={
        "architecture": "LeNet",
        "epochs": EPOCHS,
        "image_size": IMG_SIZE,
    }
)

experiment_name = f"lenet"

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
)

run.finish()
wandb.join()

del train_loader
del test_loader

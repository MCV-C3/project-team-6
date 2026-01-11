import torch
from augmentation import make_full_augmentation
import utils
import wandb

from models.base import WraperModel
from pipeline import experiment


argparser = utils.get_experiment_argument_parser()
args = argparser.parse_args()

EPOCHS = args.epochs
IMG_SIZE = 224

dry = args.dry
device = utils.set_device(args.gpu_id)

# se le puede pasar train_folder y test_folder para cambiar el dataset
train_loader, test_loader = utils.get_loaders(image_size=(IMG_SIZE, IMG_SIZE), 
                                              resize_train=True, 
                                              resize_test=True, 
                                              train_batch_size=64, 
                                              train_folder="~/mcv/datasets/C3/2425/MIT_large_train/train", 
                                              test_folder="~/mcv/datasets/C3/2425/MIT_large_train/test")

model = WraperModel(8, feature_extraction=True)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.0001)

augmentation = None # make_full_augmentation((IMG_SIZE, IMG_SIZE))

run_name = f"Example"

run = utils.init_wandb_run(
    dry=dry,
    entity="mcv-team-6",
    project="C3-Week3",
    name=run_name,
    config={
        "architecture": "Example",
        "epochs": EPOCHS,
        "image_size": IMG_SIZE,
    }
)

experiment_name = f"example"

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

import torch
from augmentation import make_full_augmentation
import utils
import wandb

from models.descriptor_classifier import make_from_widths
from pipeline import experiment
from models.base_model import BaseModel


argparser = utils.get_experiment_argument_parser()
argparser.add_argument('--depth', type=int, default=4, help='Depth of the hidden layer')
argparser.add_argument('--width', type=int, default=300, help='Width of the hidden layer')
args = argparser.parse_args()

EPOCHS = args.epochs
DEPTH = args.depth
WIDTH = args.width
IMG_SIZE = 224

device = utils.set_device(args.gpu_id)

train_loader, test_loader = utils.get_loaders(image_size=(IMG_SIZE, IMG_SIZE))

input_d = 3*IMG_SIZE*IMG_SIZE
descriptor_widths = [WIDTH]*DEPTH #Assume all hidden layers same size
classifier_widths = [11]


model = make_from_widths(input_d=input_d, descriptor_widths=descriptor_widths, classifier_widths=classifier_widths)

# print(model)

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

experiment(f"base_model_depth_{DEPTH}",
    model=model,
    optimizer=optimizer,
    criterion=loss,
    epochs=EPOCHS,
    train_loader=train_loader,
    test_loader=test_loader,
    augmentation=make_full_augmentation((IMG_SIZE, IMG_SIZE)),
    wandb_run=run,
    device=device,
)

run.finish()
wandb.join()

del train_loader
del test_loader
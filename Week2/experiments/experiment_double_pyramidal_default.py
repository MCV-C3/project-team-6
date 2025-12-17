import torch
from augmentation import make_full_augmentation
import utils
import wandb

from models.complex_models.pyramidal_dense_descriptor_classifier_v2 import make_double_descriptor_prototype
from pipeline import experiment


argparser = utils.get_experiment_argument_parser()
argparser.add_argument('--dropout', type=float, default=0, help='Dropout probability')
argparser.add_argument('--weight-decay', type=float, default=0, help='Weight decay (L2 regularization)')
argparser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing')
args = argparser.parse_args()

EPOCHS = args.epochs
IMG_SIZE = 224
DROPOUT = args.dropout
WEIGHT_DECAY = args.weight_decay
LABEL_SMOOTHING = args.label_smoothing

dry = args.dry
device = utils.set_device(args.gpu_id)

train_loader, test_loader = utils.get_loaders(image_size=(IMG_SIZE, IMG_SIZE), resize_train=False)

model = make_double_descriptor_prototype(dropout=DROPOUT)
loss = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=WEIGHT_DECAY)

augmentation = make_full_augmentation((IMG_SIZE, IMG_SIZE))

run = utils.init_wandb_run(
    dry=dry,
    entity="mcv-team-6",
    project="C3-Week2",
    name=f"Pyramidal Double Default (dropout={DROPOUT} wd={WEIGHT_DECAY} ls={LABEL_SMOOTHING})",
    config={
        "architecture": "PyramidalDoubleDefault",
        "patch_structure": "16x16 -> 7x7 -> 2x2",
        "channels": "3 -> 128 -> 1024 -> 4096",
        "epochs": EPOCHS,
        "image_size": IMG_SIZE,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
    }
)

experiment(
    f"pyramidal_double_default",
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

import torch
from augmentation import make_full_augmentation
import utils
import wandb

from models.patch_based_classifier import make_patch_model
from pipeline import experiment


argparser = utils.get_experiment_argument_parser()
argparser.add_argument('--patch-size', type=int, default=8, help='Size of the square patches')
argparser.add_argument('--stride', type=int, default=None, help='Stride for patch extraction (default: same as patch_size for non-overlapping)')
argparser.add_argument('--depth', type=int, default=2, help='Depth of the descriptor MLP')
argparser.add_argument('--width', type=int, default=512, help='Width of the first hidden layer in descriptor MLP')
argparser.add_argument('--merge-strategy', type=str, default='mean',
                       choices=['mean', 'max', 'median', 'attention', 'voting'],
                       help='Strategy to merge patch predictions')
args = argparser.parse_args()

EPOCHS = args.epochs
PATCH_SIZE = args.patch_size
STRIDE = args.stride if args.stride is not None else PATCH_SIZE
DEPTH = args.depth
WIDTH = args.width
MERGE_STRATEGY = args.merge_strategy
IMG_SIZE = 224

dry = args.dry
device = utils.set_device(args.gpu_id)

train_loader, test_loader = utils.get_loaders(image_size=(IMG_SIZE, IMG_SIZE), resize_train=False)

# descriptor_widths = [WIDTH]
# for i in range(1, DEPTH):
#     descriptor_widths.append(max(128, WIDTH // (2 ** i)))

descriptor_widths = [WIDTH] * DEPTH

model = make_patch_model(
    input_channels=3,
    patch_size=PATCH_SIZE,
    stride=STRIDE,
    descriptor_widths=descriptor_widths,
    num_classes=11,
    merge_strategy=MERGE_STRATEGY,
)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 0.0001)

augmentation = make_full_augmentation((IMG_SIZE, IMG_SIZE))

overlap_str = f"stride={STRIDE}" if STRIDE != PATCH_SIZE else "non-overlap"
run_name = f"PatchModel: patch={PATCH_SIZE} {overlap_str} merge={MERGE_STRATEGY} w={WIDTH} d={DEPTH}"

run = utils.init_wandb_run(
    dry=dry,
    entity="mcv-team-6",
    project="C3-Week2",
    name=run_name,
    config={
        "architecture": "PatchBasedClassifier",
        "patch_size": PATCH_SIZE,
        "stride": STRIDE,
        "merge_strategy": MERGE_STRATEGY,
        "descriptor_widths": descriptor_widths,
        "epochs": EPOCHS,
        "image_size": IMG_SIZE,
    }
)

experiment_name = f"patch_model_ps{PATCH_SIZE}_s{STRIDE}_{MERGE_STRATEGY}_w{WIDTH}_d{DEPTH}"

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

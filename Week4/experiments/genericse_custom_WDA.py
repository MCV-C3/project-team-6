from trainers.basic_trainer import BasicTrainingModule
from models.generic_model import GenericSE
import utils
import torch
from pytorch_lightning.loggers import WandbLogger
import kornia.augmentation as ka

argparser = utils.get_experiment_argument_parser()
args = argparser.parse_args()

EPOCHS = args.epochs
IMG_SIZE = 224
LR = 0.001

device = utils.set_device(args.gpu_id)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

augmentation = ka.AugmentationSequential(
    ka.RandomHorizontalFlip(p=0.5),
    ka.RandomRotation(degrees=15, p=0.2),
    ka.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    ka.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.3),
    ka.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=0.3),
    ka.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3),
    ka.RandomGrayscale(p=0.1),
    ka.Resize(size=(IMG_SIZE, IMG_SIZE))
)

# 3. Definición del Modelo Ganador (GenericSE)
# Config: layers_8:0, layers_16:1, layers_24:0, layers_32:4
# Resulting Channels: [16, 24, 24, 24, 32, 32, 32, 32]
best_channels_config = [16] + [24]*3 + [32]*4

model = GenericSE(channels=best_channels_config, num_class=8)
total_params = sum(p.numel() for p in model.parameters())

# 4. Módulo de Entrenamiento
train_model = BasicTrainingModule(model=model, augmentations=augmentation, lr=LR)

# 5. Logger de WandB
wandb_logger = WandbLogger(
    project="C3-Week4",
    entity="mcv-team-6",
    name="GenericSE Best Sweep DA custom1"
)

wandb_logger.experiment.config.update({
    "architecture": "GenericSE",
    "configuration": "Best Sweep (0-1-3-4)",
    "channels": best_channels_config,
    "augmentation": "Custom DA",
    "epochs": EPOCHS,
    "image_size": IMG_SIZE,
    "learning_rate": LR,
    "parameters": total_params
})

# 6. Trainer y Loaders
# Usamos patience=40 como en el experimento de densenet_WDA
trainer = utils.get_trainer(wandb_logger, patience=40, min_delta=0.001, epochs=EPOCHS)

train_loader, test_loader = utils.get_loaders(
    image_size=(IMG_SIZE, IMG_SIZE),
    resize_train=True,
    resize_test=True,
    train_batch_size=64,
    train_folder=args.train_folder,
    test_folder=args.test_folder,
)

# 7. Ejecución
trainer.fit(train_model, train_loader, test_loader)
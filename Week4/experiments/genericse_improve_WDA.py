from trainers.basic_trainer import BasicTrainingModule, BasicTrainingModuleSchedule
from trainers.knowledge_trainer import DistillationTrainingModule
from models.generic_model import GenericSE, GenericSEHandmade
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
        ka.RandomHorizontalFlip(p=0.04855768013650663),
        ka.RandomRotation(degrees=16,
                          p=0.13009412985538538),
        ka.RandomAffine(degrees=0, shear=29,
                            translate=(0.3584182906605954, 0.035543949656199385),
                            scale=(0.999217703073204, 0.999217703073204)),
        ka.RandomResizedCrop(size=(IMG_SIZE,IMG_SIZE), scale=(0.4382924444152586, 1.0), ratio=(1.0, 1.0), p=0.1939592095722116),
        ka.RandomGaussianBlur(kernel_size=(9, 9), sigma=(1.190437323963532, 1.190437323963532), p=0.442038986973481),
        ka.ColorJitter(brightness=1.1237698676931558,
                           contrast=0.7852337555662053,
                           saturation=1.20141373248246,
                           hue=0.06272478548259297,
                           p=0.2029549731865357),
        ka.RandomGrayscale(p=0.331441327268726),
        ka.Resize(size=(IMG_SIZE, IMG_SIZE))
    )

# 3. Definición del Modelo Ganador (GenericSE)
# Config: layers_8:0, layers_16:1, layers_24:0, layers_32:5
best_channels_config = [16] + [24]*3 + [32]*4

model = GenericSEHandmade(num_class=8)
total_params = sum(p.numel() for p in model.parameters())

# 4. Módulo de Entrenamiento
train_model = DistillationTrainingModule(student=model, augmentations=augmentation, lr=LR)

# 5. Logger de WandB
wandb_logger = WandbLogger(
    project="C3-Week4",
    entity="mcv-team-6",
    name="GenericSE DA KA Handmade -dropout"
)

wandb_logger.experiment.config.update({
    "architecture": "GenericSE",
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
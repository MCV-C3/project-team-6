from trainers.basic_trainer import BasicTrainingModule
from models.micro_densenet import MicroDenseNet
import utils
import torch
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import kornia.augmentation as ka


argparser = utils.get_experiment_argument_parser()
args = argparser.parse_args()

EPOCHS = args.epochs
IMG_SIZE = 224
LR = 0.001

dry = args.dry
device = utils.set_device(args.gpu_id)

torch.manual_seed(0)
torch.cuda.manual_seed(0)

augmentation = ka.AugmentationSequential(
        ka.RandomHorizontalFlip(p=0.5),
        ka.RandomRotation(degrees=15, p=0.2),
        ka.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        ka.RandomResizedCrop(size=(IMG_SIZE,IMG_SIZE), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.3),
        ka.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=0.3),
        ka.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3),
        ka.RandomGrayscale(p=0.1),
        ka.Resize(size=(IMG_SIZE, IMG_SIZE))
    )

model = MicroDenseNet(in_channels=3, num_classes=8, growth_rate=6, block_layers=4)
total_params = sum(p.numel() for p in model.parameters())
train_model = BasicTrainingModule(model=model, augmentations=augmentation, lr=LR)

wandb_logger = WandbLogger(
    project="C3-Week4",
    entity="mcv-team-6",
    name=f"MicroDenseNet DA Standard Numbers"
)

wandb_logger.experiment.config.update({
                "architecture": "Micro DenseNet Depthwise",
                "epochs": EPOCHS,
                "image_size": IMG_SIZE,
                "learning_rate": LR,
                "parameters" : total_params,
                "growth_rate": 6,
                "layers_per_block": 4
            })

trainer = utils.get_trainer(wandb_logger, patience=40, min_delta=0.001, epochs=EPOCHS)

train_loader, test_loader = utils.get_loaders(
    image_size=(IMG_SIZE, IMG_SIZE),
    resize_train=True,
    resize_test=True,
    train_batch_size=64,
    train_folder=args.train_folder,
    test_folder=args.test_folder,
)

trainer.fit(train_model, train_loader, test_loader)
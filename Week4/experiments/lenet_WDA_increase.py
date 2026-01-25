from trainers.basic_trainer import BasicTrainingModule
from trainers.knowledge_trainer import DistillationTrainingModule
from models.small_lenet import SmallLeNetSE
import utils
import torch
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import kornia.augmentation as ka


argparser = utils.get_experiment_argument_parser()
args = argparser.parse_args()

EPOCHS = args.epochs
IMG_SIZE = 224
LR = 0.0001

dry = args.dry
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

model = SmallLeNetSE()
total = sum(p.numel() for p in model.parameters())
train_model = DistillationTrainingModule(student=model, augmentations=augmentation, lr=0.001)

wandb_logger = WandbLogger(
    project="C3-Week4",
    entity="mcv-team-6",
    name="Small LeNetSE WDA KD"
)

wandb_logger.experiment.config.update({
                "architecture": "Small LeNet SE",
                "epochs": EPOCHS,
                "image_size": IMG_SIZE,
                "learning_rate": LR,
                "parameters" : total
            })

trainer = utils.get_trainer(wandb_logger, patience=30, min_delta=0.001, epochs=EPOCHS, model_name="small_lenet_se")

train_loader, test_loader = utils.get_loaders(
    image_size=(IMG_SIZE, IMG_SIZE),
    resize_train=True,
    resize_test=True,
    train_batch_size=64,
    train_folder=args.train_folder,
    test_folder=args.test_folder,
)

trainer.fit(train_model, train_loader, test_loader)
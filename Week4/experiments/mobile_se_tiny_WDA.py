from trainers.basic_trainer import BasicTrainingModule
from models.small_mobile_se import TinyMobileSE2k, TinyMobileSEStartBig, TinyMobileSEExtended
import utils
import torch
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import kornia.augmentation as ka
import wandb

argparser = utils.get_experiment_argument_parser()
args = argparser.parse_args()

EPOCHS = args.epochs
IMG_SIZE = 224
LR = 0.001

dry = args.dry
device = utils.set_device(args.gpu_id)

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

torch.manual_seed(0)
torch.cuda.manual_seed(0)

for m in [TinyMobileSE2k, TinyMobileSEStartBig, TinyMobileSEExtended]:
    # Close any previous run BEFORE creating a new WandbLogger
    wandb.finish()

    model = m()
    total = sum(p.numel() for p in model.parameters())
    train_model = BasicTrainingModule(model=model, augmentations=augmentation, lr=LR)

    # Put config here so it is set at wandb.init time (no later mutation)
    wandb_logger = WandbLogger(
        project="C3-Week4",
        entity="mcv-team-6",
        name=f"SmallMobileSE/{m.__name__} Before ReLU",
        group="SmallMobileSE_Ablation",
        config={
            "architecture": m.__name__,
            "epochs": EPOCHS,
            "image_size": IMG_SIZE,
            "learning_rate": LR,
            "parameters": total
        },
    )

    # IMPORTANT: don't call wandb_logger.experiment.config.update(...)
    # because Lightning may reuse a run and W&B blocks changing existing keys.

    trainer = utils.get_trainer(wandb_logger, patience=30, min_delta=0.001, epochs=EPOCHS, model_name=m.__name__)

    train_loader, test_loader = utils.get_loaders(
        image_size=(IMG_SIZE, IMG_SIZE),
        resize_train=True,
        resize_test=True,
        train_batch_size=64,
        train_folder=args.train_folder,
        test_folder=args.test_folder,
    )

    trainer.fit(train_model, train_loader, test_loader)

    # Finish the run created/used by this logger (more reliable than plain wandb.finish())
    wandb_logger.experiment.finish()

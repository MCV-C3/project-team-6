from trainers.wd_trainer import WeightDecayTrainer
from models.variable_generic_model import VariableGenericSE
import utils
import torch
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import kornia.augmentation as ka
import wandb


argparser = utils.get_experiment_argument_parser()
args, unknown = argparser.parse_known_args()
sweep_config = {}
for arg in unknown:
    if arg.startswith("--"):
        key, val = arg[2:].split("=")
        try:
            val = float(val)
            if val.is_integer():
                val = int(val)
        except ValueError:
            pass
        sweep_config[key] = val

EPOCHS = args.epochs
IMG_SIZE = 224
LR = 0.001

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

wandb_logger = WandbLogger(
    project="C3-Week4",
    entity="mcv-team-6",
    name="Reg Search"
)

wandb_logger.experiment.config.update(sweep_config)
cfg = wandb_logger.experiment.config


start_layer = 0 if cfg.get('dropout_loc', 'mid') == "all" else 3

model = VariableGenericSE(
    channels_list=[16, 24, 24, 24, 32, 32, 32, 32], 
    use_residuals=cfg.get('use_residuals', False),
    use_cbam=cfg.get('use_cbam', False),
    dropout_rate=cfg.get('dropout_val', 0.0),
    dropout_start_layer=start_layer
)


total_params = sum(p.numel() for p in model.parameters())
sweep_config["parameters"] = total_params
wandb_logger.experiment.config.update(sweep_config)



train_model = WeightDecayTrainer(model=model, augmentations=augmentation, lr=LR, weight_decay=cfg.get('weight_decay', 0.0))




trainer = utils.get_trainer(wandb_logger, patience=30, min_delta=0.001, epochs=EPOCHS)

train_loader, test_loader = utils.get_loaders(
    image_size=(IMG_SIZE, IMG_SIZE),
    resize_train=True,
    resize_test=True,
    train_batch_size=64,
    train_folder=args.train_folder,
    test_folder=args.test_folder,
)

trainer.fit(train_model, train_loader, test_loader)

wandb_logger.experiment.finish()
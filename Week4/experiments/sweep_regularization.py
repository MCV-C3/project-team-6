from trainers.wd_trainer import WeightDecayTrainer
from models.variable_generic_model import VariableGenericSE
import utils
import torch
import torch.nn as nn
from pytorch_lightning.loggers import WandbLogger
import kornia.augmentation as ka
import wandb

argparser = utils.get_experiment_argument_parser()
args, unknown = argparser.parse_known_args()
wandb.init()
config = wandb.config

start_layer = 0 if config.dropout_loc == "all" else 3

model = VariableGenericSE(
    channels_list=[16, 24, 24, 24, 32, 32, 32, 32], 
    use_residuals=config.use_residuals,
    use_cbam=config.use_cbam,
    dropout_rate=config.dropout_val,
    dropout_start_layer=start_layer
)

IMG_SIZE = 224
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

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

train_model = WeightDecayTrainer(
    model=model, 
    loss_fn=loss_fn,
    augmentations=augmentation, 
    lr=0.001,
    weight_decay=config.weight_decay
)


wandb_logger = WandbLogger(project="C3-Week4", entity="mcv-team-6")
wandb_logger.experiment.config.update({
    "parameters": sum(p.numel() for p in model.parameters()),
    "architecture": "GenericSE_Reg_Search"
})

trainer = utils.get_trainer(wandb_logger, patience=30, min_delta=0.001, epochs=args.epochs)
train_loader, test_loader = utils.get_loaders(
    image_size=(IMG_SIZE, IMG_SIZE),
    resize_train=True, resize_test=True,
    train_batch_size=64,
    train_folder=args.train_folder, test_folder=args.test_folder
)

trainer.fit(train_model, train_loader, test_loader)
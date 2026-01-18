from trainers.basic_trainer import BasicTrainingModule
from models.small_lenet import SmallLeNet
import utils
import torch
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl


argparser = utils.get_experiment_argument_parser()
args = argparser.parse_args()

EPOCHS = args.epochs
IMG_SIZE = 224
LR = 0.0001

dry = args.dry
device = utils.set_device(args.gpu_id)

torch.manual_seed(0)
torch.cuda.manual_seed(0)

model = SmallLeNet()
total = sum(p.numel() for p in model.parameters())
train_model = BasicTrainingModule(model=model)

wandb_logger = WandbLogger(
    project="C3-Week4",
    entity="mcv-team-6",
    name="Test all epochs"
)

wandb_logger.experiment.config.update({
                "architecture": "Small LeNet",
                "epochs": EPOCHS,
                "image_size": IMG_SIZE,
                "learning_rate": LR,
                "parameters" : total
            })

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

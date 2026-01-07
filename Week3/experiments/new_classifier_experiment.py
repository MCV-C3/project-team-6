from copy import deepcopy
import torch
import torch.nn as nn
from torchvision import models

import utils
from models.base import WraperModel
from pipeline import experiment

import wandb


def set_classifier(model, head):
    model.backbone.classifier = head
    return model

def create_head_n_classifier_layers(in_features, list_of_neurons: list, num_classes: int):
    if not list_of_neurons:
        return nn.Linear(in_features, num_classes)

    layers = []
    prev_in = in_features
    n = len(list_of_neurons)

    for i in range(n):
        out_features = list_of_neurons[i]
        layers.append(nn.Linear(prev_in, out_features))
        layers.append(nn.ReLU())
        prev_in = out_features

    layers.append(nn.Linear(prev_in, num_classes))
    return nn.Sequential(*layers)


argparser = utils.get_experiment_argument_parser()
args = argparser.parse_args()

#argparser.add_argument('--weight-decay', type=float, default=0, help='Weight decay (L2 regularization)')
#argparser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing')

EPOCHS = args.epochs
IMG_SIZE = 224
LR = 0.0001
WEIGHT_DECAY = 1e-5 #args.weight_decay
LABEL_SMOOTHING = 0.1 #args.label_smoothing

dry = args.dry
device = utils.set_device(args.gpu_id)

train_loader, test_loader = utils.get_loaders(
    image_size=(IMG_SIZE, IMG_SIZE),
    resize_train=True,
    resize_test=True,
    train_batch_size=64,
    train_folder="~/mcv/datasets/C3/2425/MIT_large_train/train",
    test_folder="~/mcv/datasets/C3/2425/MIT_large_train/test"
)

model = WraperModel(num_classes=8, feature_extraction=True)
in_features = model.backbone.classifier.in_features
loss = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

for neurons in [[512],[1024],[2048], [1024, 512], [2048, 1024, 512]]:
    head = create_head_n_classifier_layers(in_features, neurons, 8)
    id_exp = "new_model_classifier_" + "x".join(str(l.out_features) for l in head if isinstance(l, nn.Linear) and l.out_features != 8)
    model_copy = deepcopy(model)
    model_copy = set_classifier(model_copy, head)
    model_copy = model_copy.to(device)
    for p in model_copy.backbone.classifier.parameters():
        p.requires_grad = True

    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_copy.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY)

    run = utils.init_wandb_run(
        dry=dry,
        entity="mcv-team-6",
        project="C3-Week3",
        name=id_exp,
        config={
            "architecture": "Densenet121-NewHead",
            "head_structure": id_exp.replace("new_model_classifier_", ""),
            "epochs": EPOCHS,
            "image_size": IMG_SIZE,
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
        }
    )

    experiment(
        f"new_model_classifiers",
        model=model_copy,
        optimizer=optimizer,
        criterion=loss,
        epochs=EPOCHS,
        train_loader=train_loader,
        test_loader=test_loader,
        wandb_run=run,
        device=device,
    )

    run.finish()
    del model_copy
    del optimizer
    torch.cuda.empty_cache()

wandb.join()
del train_loader
del test_loader
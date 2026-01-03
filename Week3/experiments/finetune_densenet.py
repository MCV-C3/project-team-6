import torch
from augmentation import make_full_augmentation
import utils
import wandb
import copy

from models.base import WraperModel
from pipeline import experiment, multi_stage_experiment


def get_densenet_layers_to_unfreeze(model):
    """
    Returns a list of layer groups to progressively unfreeze.
    Order: classifier -> denseblock4 (layer by layer) -> denseblock3 (layer by layer)
    """
    base_model = model.model
    layers_to_unfreeze = []
    
    # First stage: only classifier (already done by feature_extraction=True)
    layers_to_unfreeze.append([])
    
    # Get all layers from denseblock4 in reverse order
    denseblock4_layers = []
    for name, module in base_model.features.denseblock4.named_children():
        denseblock4_layers.append(('features.denseblock4.' + name, module))
    
    # Add each layer of denseblock4 individually (reverse order)
    for layer_name, layer_module in reversed(denseblock4_layers):
        layers_to_unfreeze.append([(layer_name, layer_module)])
    
    # Get all layers from denseblock3 in reverse order
    denseblock3_layers = []
    for name, module in base_model.features.denseblock3.named_children():
        denseblock3_layers.append(('features.denseblock3.' + name, module))
    
    # Add each layer of denseblock3 individually (reverse order)
    for layer_name, layer_module in reversed(denseblock3_layers):
        layers_to_unfreeze.append([(layer_name, layer_module)])
    
    return layers_to_unfreeze


def freeze_all_except_classifier(model):
    """Freeze all layers except the classifier."""
    for param in model.model.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier
    for param in model.model.classifier.parameters():
        param.requires_grad = True


def unfreeze_layers(layers_list):
    """Unfreeze specific layers."""
    for layer_name, layer_module in layers_list:
        for param in layer_module.parameters():
            param.requires_grad = True


argparser = utils.get_experiment_argument_parser()
args = argparser.parse_args()

EPOCHS = args.epochs
IMG_SIZE = 224

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

# Initialize model with feature extraction (only classifier unfrozen)
model = WraperModel(8, feature_extraction=True)

loss = torch.nn.CrossEntropyLoss()
augmentation = None

run_name = f"FineTune_DenseNet121_Progressive"

run = utils.init_wandb_run(
    dry=dry,
    entity="mcv-team-6",
    project="C3-Week3",
    name=run_name,
    config={
        "architecture": "DenseNet-121-Progressive-FineTune",
        "epochs_per_stage": EPOCHS,
        "image_size": IMG_SIZE,
        "strategy": "progressive_layer_unfreezing",
    }
)

# Get layers to progressively unfreeze
layers_to_unfreeze = get_densenet_layers_to_unfreeze(model)

experiment_name = f"finetune_densenet121_progressive"

# Multi-stage experiment
stages_config = []

for stage_idx, layer_group in enumerate(layers_to_unfreeze):
    if stage_idx == 0:
        stage_name = "Stage_0_Classifier_Only"
        freeze_all_except_classifier(model)
    else:
        layer_names = [name for name, _ in layer_group]
        stage_name = f"Stage_{stage_idx}_Unfreeze_{layer_names[0] if layer_names else 'unknown'}"
        unfreeze_layers(layer_group)
    
    # Create new optimizer for each stage with current trainable parameters
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 0.0001)
    
    stages_config.append({
        "stage_name": stage_name,
        "model": model,
        "optimizer": optimizer,
        "criterion": loss,
        "epochs": EPOCHS,
    })

multi_stage_experiment(
    experiment_name,
    stages_config=stages_config,
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

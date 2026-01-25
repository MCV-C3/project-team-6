from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision import models
from models.squeeze_excitation import SqueezeExcitation
from augmentation import make_full_augmentation
import utils
import wandb
from copy import deepcopy

from models.base import WraperModel


def get_denseblock_channels(denseblock):
    num_layers = len(denseblock)
    first_layer = getattr(denseblock, 'denselayer1')
    growth_rate = first_layer.conv2.out_channels
    input_channels = first_layer.norm1.num_features
    
    return input_channels + num_layers * growth_rate

def add_squeeze_excitation_blocks(backbone: models.densenet.DenseNet):
    new_features = OrderedDict()
    
    for name, module in backbone.features.named_children():
        new_features[name] = module
        
        if name.startswith('denseblock'):
            channels = get_denseblock_channels(module)
            se_name = f"se{name[-1]}"
            new_features[se_name] = SqueezeExcitation(channels)
    
    backbone.features = nn.Sequential(new_features)
    backbone.classifier = nn.Linear(1024, 8)
    
    return backbone

def unfreeze_stage1(backbone: models.densenet.DenseNet):
    """
    Unfreeze denseblock4, se4, norm5, and classifier.
    """
    layers_to_unfreeze = ['denseblock4', 'se4', 'norm5']

    for name, module in backbone.features.named_children():
        if name in layers_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True

    # Unfreeze classifier
    for param in backbone.classifier.parameters():
        param.requires_grad = True

    return backbone

def unfreeze_stage2(backbone: models.densenet.DenseNet):
    """
    Unfreeze denseblock3, se3, transition3, denseblock4, se4, norm5, and classifier.
    """
    layers_to_unfreeze = ['denseblock3', 'se3', 'transition3', 'denseblock4', 'se4', 'norm5']

    for name, module in backbone.features.named_children():
        if name in layers_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True

    # Unfreeze classifier
    for param in backbone.classifier.parameters():
        param.requires_grad = True

    return backbone

def unfreeze_stage3(backbone: models.densenet.DenseNet):
    """
    Unfreeze all layers in the backbone.
    """
    for param in backbone.parameters():
        param.requires_grad = True

    return backbone

def create_model_for_experiment(experiment_type, num_classes=8):
    model = WraperModel(num_classes=num_classes, feature_extraction=False)

    model.modify_layers(add_squeeze_excitation_blocks)
    
    return model

import torch
import torch.nn as nn
from torchvision import models
import utils
import wandb
from copy import deepcopy

from models.base import WraperModel
from pipeline import experiment


def create_denseblock(num_layers, in_channels, growth_rate=32, bn_size=4):
    """
    Create a DenseNet-style dense block.

    Args:
        num_layers: Number of dense layers in the block
        in_channels: Number of input channels
        growth_rate: How many filters to add each layer (k in the paper)
        bn_size: Multiplicative factor for bottleneck layers
    """
    from torchvision.models.densenet import _DenseBlock
    return _DenseBlock(
        num_layers=num_layers,
        num_input_features=in_channels,
        bn_size=bn_size,
        growth_rate=growth_rate,
        drop_rate=0
    )


def remove_last_block(backbone):
    """
    Remove denseblock4 + transition3.
    Architecture becomes: ... -> denseblock3 -> norm5 -> avgpool -> classifier
    Output features from denseblock3: 1024 channels
    """
    new_features = []
    for name, module in backbone.features.named_children():
        if name in ['denseblock4', 'transition3']:
            continue
        elif name == 'norm5':
            new_features.append(nn.BatchNorm2d(1024))
        else:
            new_features.append(module)

    backbone.features = nn.Sequential(*new_features)
    backbone.classifier = nn.Linear(1024, backbone.classifier.out_features)

    return backbone


def remove_last_two_blocks(backbone):
    """
    Remove denseblock4 + transition3 + denseblock3 + transition2.
    Architecture becomes: ... -> denseblock2 -> norm5 -> avgpool -> classifier
    Output features from denseblock2: 512 channels
    """
    new_features = []
    for name, module in backbone.features.named_children():
        if name in ['denseblock4', 'transition3', 'denseblock3', 'transition2']:
            continue
        elif name == 'norm5':
            new_features.append(nn.BatchNorm2d(512))
        else:
            new_features.append(module)

    backbone.features = nn.Sequential(*new_features)
    backbone.classifier = nn.Linear(512, backbone.classifier.out_features)

    return backbone


def add_new_block(backbone):
    """
    Add a new dense block after denseblock4 (before classifier).
    Architecture becomes: ... -> denseblock4 -> norm5 -> new_denseblock5 -> new_norm -> avgpool -> classifier

    The new block receives 1024 channels from denseblock4 and adds 16 layers with growth_rate=32,
    resulting in 1024 + 16*32 = 1536 output channels.
    """
    new_dense_block = create_denseblock(
        num_layers=16,
        in_channels=1024,
        growth_rate=32,
        bn_size=4
    )
    new_norm = nn.BatchNorm2d(1536)

    new_features = []
    for name, module in backbone.features.named_children():
        new_features.append(module)

    new_features.append(new_dense_block)
    new_features.append(new_norm)

    backbone.features = nn.Sequential(*new_features)
    backbone.classifier = nn.Linear(1536, backbone.classifier.out_features)

    return backbone


def freeze_pretrained_layers(model, unfreeze_new_block=False):
    """
    Freeze all pretrained layers, only train classifier and new/modified blocks.

    Args:
        model: The WraperModel instance
        unfreeze_new_block: If True, unfreeze the newly added block (for add_new_block experiment)
    """
    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.backbone.classifier.parameters():
        param.requires_grad = True

    if unfreeze_new_block:
        if len(list(model.backbone.features.children())) > 10:
            new_block = list(model.backbone.features.children())[-2]
            new_norm = list(model.backbone.features.children())[-1]

            for param in new_block.parameters():
                param.requires_grad = True
            for param in new_norm.parameters():
                param.requires_grad = True


def create_model_for_experiment(experiment_type, num_classes=8):
    """
    Create a modified DenseNet model for the specified experiment.

    Args:
        experiment_type: One of 'remove_1_block', 'remove_2_blocks', 'add_block'
        num_classes: Number of output classes
    """
    model = WraperModel(num_classes=num_classes, feature_extraction=False)

    if experiment_type == 'remove_1_block':
        model.modify_layers(remove_last_block)
        freeze_pretrained_layers(model, unfreeze_new_block=False)
    elif experiment_type == 'remove_2_blocks':
        model.modify_layers(remove_last_two_blocks)
        freeze_pretrained_layers(model, unfreeze_new_block=False)
    elif experiment_type == 'add_block':
        model.modify_layers(add_new_block)
        freeze_pretrained_layers(model, unfreeze_new_block=True)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    return model


if __name__ == "__main__":
    argparser = utils.get_experiment_argument_parser()
    args = argparser.parse_args()

    EPOCHS = args.epochs
    IMG_SIZE = 224
    LR = 0.0001

    dry = args.dry
    device = utils.set_device(args.gpu_id)

    train_loader, test_loader = utils.get_loaders(
        image_size=(IMG_SIZE, IMG_SIZE),
        resize_train=True,
        resize_test=True,
        train_batch_size=64,
        train_folder="/home/arnau-marcos-almansa/workspace/C3/data/MIT_large_train/train",
        test_folder="/home/arnau-marcos-almansa/workspace/C3/data/MIT_large_train/test"
    )

    loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    experiments_config = [
        {
            'type': 'remove_1_block',
            'name': 'Remove Last Block (DB4+T3)',
            'wandb_name': 'Remove-1-Block'
        },
        {
            'type': 'remove_2_blocks',
            'name': 'Remove Last 2 Blocks (DB4+T3+DB3+T2)',
            'wandb_name': 'Remove-2-Blocks'
        },
        {
            'type': 'add_block',
            'name': 'Add New Block (DB5 after DB4)',
            'wandb_name': 'Add-New-Block'
        }
    ]

    num_experiments = len(experiments_config)
    print(f"Will run {num_experiments} block modification experiments\n")

    for exp_idx, exp_config in enumerate(experiments_config):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {exp_idx + 1}/{num_experiments}")
        print(f"{exp_config['name']}")
        print(f"{'='*80}\n")

        model = create_model_for_experiment(exp_config['type'], num_classes=8)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        print("\nModel architecture:")
        print(f"  Features: {len(list(model.backbone.features.children()))} layers")
        print(f"  Classifier input: {model.backbone.classifier.in_features} features")
        print(f"  Classifier output: {model.backbone.classifier.out_features} classes")

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=1e-5)

        run = utils.init_wandb_run(
            dry=dry,
            entity="mcv-team-6",
            project="C3-Week3",
            name=exp_config['wandb_name'],
            config={
                "architecture": f"DenseNet121-{exp_config['type']}",
                "experiment_type": exp_config['type'],
                "epochs": EPOCHS,
                "image_size": IMG_SIZE,
                "learning_rate": LR,
                "trainable_params": trainable,
                "total_params": total,
                "trainable_percentage": 100*trainable/total,
                "classifier_input_features": model.backbone.classifier.in_features,
            }
        )

        experiment_name = f"block_mod_{exp_config['type']}"
        experiment(
            experiment_name,
            model=model,
            optimizer=optimizer,
            criterion=loss,
            epochs=EPOCHS,
            train_loader=train_loader,
            test_loader=test_loader,
            augmentation=None,
            wandb_run=run,
            device=device,
            early_stopping_patience=15,
            early_stopping_min_delta=0.001,
        )

        run.finish()

        del model
        del optimizer
        torch.cuda.empty_cache()

        print(f"\nCompleted experiment {exp_idx + 1}/{num_experiments}\n")

    wandb.join()

    del train_loader
    del test_loader

    print(f"\nAll block modification experiments completed!")
    print(f"Total experiments run: {num_experiments}")

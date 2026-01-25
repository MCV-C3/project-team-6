from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision import models
from models.squeeze_escitation import SqueezeExcitation
from augmentation import make_full_augmentation
import utils
import wandb
from copy import deepcopy

from models.base import WraperModel
from pipeline import experiment


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
        train_folder=args.train_folder,
        test_folder=args.test_folder,
    )

    loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    experiments_config = [
        {
            'type': 'add_squeeze_excitation',
            'name': 'Add Squeeze and Excitation after each denseblock',
            'wandb_name': 'Squeeze-and-Excitation Unfreezed'
        },
    ]

    augmentation = None
    
    if args.augmentation:
        print("Added data augmentation")
        augmentation = make_full_augmentation((IMG_SIZE, IMG_SIZE))

    num_experiments = len(experiments_config)
    print(f"Will run {num_experiments} Squeeze and Excitation experiments\n")

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

        experiment_name = f"squeeze_and_excitation_{exp_config['type']}"
        experiment(
            experiment_name,
            model=model,
            optimizer=optimizer,
            criterion=loss,
            epochs=EPOCHS,
            train_loader=train_loader,
            test_loader=test_loader,
            augmentation=augmentation,
            wandb_run=run,
            device=device,
            early_stopping_patience=15,
            early_stopping_min_delta=0.001,
        )
        
        # model.modify_layers(unfreeze_stage1)

        # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        # optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=1e-5)

        # experiment(
        #     experiment_name,
        #     model=model,
        #     optimizer=optimizer,
        #     criterion=loss,
        #     epochs=EPOCHS,
        #     train_loader=train_loader,
        #     test_loader=test_loader,
        #     augmentation=None,
        #     wandb_run=run,
        #     device=device,
        #     early_stopping_patience=15,
        #     early_stopping_min_delta=0.001,
        # )
        
        # model.modify_layers(unfreeze_stage2)

        # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        # optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=1e-5)
        
        # experiment(
        #     experiment_name,
        #     model=model,
        #     optimizer=optimizer,
        #     criterion=loss,
        #     epochs=EPOCHS,
        #     train_loader=train_loader,
        #     test_loader=test_loader,
        #     augmentation=None,
        #     wandb_run=run,
        #     device=device,
        #     early_stopping_patience=15,
        #     early_stopping_min_delta=0.001,
        # )
        
        # model.modify_layers(unfreeze_stage3)
        
        # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        # optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=1e-5)
        
        # experiment(
        #     experiment_name,
        #     model=model,
        #     optimizer=optimizer,
        #     criterion=loss,
        #     epochs=EPOCHS,
        #     train_loader=train_loader,
        #     test_loader=test_loader,
        #     augmentation=None,
        #     wandb_run=run,
        #     device=device,
        #     early_stopping_patience=15,
        #     early_stopping_min_delta=0.001,
        # )
        

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

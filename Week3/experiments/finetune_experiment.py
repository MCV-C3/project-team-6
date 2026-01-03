import torch
import torch.nn as nn
from torchvision import models
import utils
import wandb

from pipeline import experiment_stages


def get_densenet_layers_to_unfreeze(model):
    """
    Get layers from DenseNet-121's last two dense blocks in reverse order.
    DenseNet-121 structure: features -> (conv0, norm0, relu0, pool0, denseblock1, transition1, 
                                          denseblock2, transition2, denseblock3, transition3, denseblock4, norm5)
    We want to unfreeze denseblock4 and denseblock3 layer by layer.
    """
    layers_to_unfreeze = []
    
    # Get denseblock4 layers (last dense block)
    denseblock4 = model.features.denseblock4
    for name, layer in reversed(list(denseblock4.named_children())):
        layers_to_unfreeze.append((f"features.denseblock4.{name}", layer))
    
    # Get transition3 (between denseblock3 and denseblock4)
    layers_to_unfreeze.append(("features.transition3", model.features.transition3))
    
    # Get denseblock3 layers
    denseblock3 = model.features.denseblock3
    for name, layer in reversed(list(denseblock3.named_children())):
        layers_to_unfreeze.append((f"features.denseblock3.{name}", layer))
    
    return layers_to_unfreeze


def freeze_all_except_classifier(model):
    """Freeze all layers except the classifier."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def create_unfreeze_callback(layers_to_unfreeze, layer_idx):
    """Create a callback that unfreezes a specific layer."""
    def callback(model, stage_idx):
        if layer_idx < len(layers_to_unfreeze):
            layer_name, layer = layers_to_unfreeze[layer_idx]
            for param in layer.parameters():
                param.requires_grad = True
            
            # Count trainable params
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"Unfroze layer: {layer_name}")
            print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return callback


def create_initial_callback():
    """Create initial callback that just prints info (classifier already unfrozen)."""
    def callback(model, stage_idx):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Initial stage: Only classifier is trainable")
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return callback


def get_optimizer_fn(lr=0.0001):
    """Return a function that creates an optimizer for trainable parameters only."""
    def optimizer_fn(model):
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        return torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-5)
    return optimizer_fn


# Main experiment
if __name__ == "__main__":
    argparser = utils.get_experiment_argument_parser()
    args = argparser.parse_args()

    EPOCHS_PER_STAGE = args.epochs
    IMG_SIZE = 224
    LR = 0.0001

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

    # Load pretrained DenseNet-121 and modify classifier
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 8)

    # Freeze all except classifier
    freeze_all_except_classifier(model)

    # Get layers to progressively unfreeze
    layers_to_unfreeze = get_densenet_layers_to_unfreeze(model)
    
    print(f"Total layers to unfreeze progressively: {len(layers_to_unfreeze)}")
    for i, (name, _) in enumerate(layers_to_unfreeze):
        print(f"  {i+1}. {name}")

    # Create stage callbacks
    # Stage 0: Train only classifier
    # Stage 1+: Unfreeze one more layer each time
    stage_callbacks = [create_initial_callback()]
    stage_names = ["Classifier Only"]
    
    for i in range(len(layers_to_unfreeze)):
        stage_callbacks.append(create_unfreeze_callback(layers_to_unfreeze, i))
        layer_name = layers_to_unfreeze[i][0].split('.')[-1]
        stage_names.append(f"+ {layer_name}")

    loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    run_name = f"Progressive-FineTuning-DenseNet121"

    run = utils.init_wandb_run(
        dry=dry,
        entity="mcv-team-6",
        project="C3-Week3",
        name=run_name,
        config={
            "architecture": "Progressive-FineTuning-DenseNet121",
            "epochs_per_stage": EPOCHS_PER_STAGE,
            "image_size": IMG_SIZE,
            "num_stages": len(stage_callbacks),
            "learning_rate": LR,
            "layers_to_unfreeze": [name for name, _ in layers_to_unfreeze],
        }
    )

    experiment_name = "progressive_finetuning_densenet121"

    all_stages_metrics = experiment_stages(
        experiment_name,
        model=model,
        optimizer_fn=get_optimizer_fn(lr=LR),
        criterion=loss,
        epochs_per_stage=EPOCHS_PER_STAGE,
        train_loader=train_loader,
        test_loader=test_loader,
        augmentation=None,
        wandb_run=run,
        stage_callbacks=stage_callbacks,
        stage_names=stage_names,
        device=device,
        early_stopping_patience=15,
        early_stopping_min_delta=0.001,
    )

    run.summary["num_stages_completed"] = len(all_stages_metrics)
    run.summary["best_stage_accuracy"] = max(s["best_test_accuracy"] for s in all_stages_metrics)
    run.summary["best_stage_loss"] = min(s["best_test_loss"] for s in all_stages_metrics)

    run.finish()
    wandb.join()

    del train_loader
    del test_loader

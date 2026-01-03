import torch
import torch.nn as nn
from torchvision import models
import utils
import wandb
from copy import deepcopy

from pipeline import experiment


def get_densenet_layers_to_unfreeze(model):
    """
    Get layers from DenseNet-121's last two dense blocks in reverse order.
    Returns layers grouped by block for step-by-step unfreezing.
    """
    layers_to_unfreeze = []
    
    # Get denseblock4 layers (last dense block) - 16 layers
    denseblock4 = model.features.denseblock4
    for name, layer in reversed(list(denseblock4.named_children())):
        layers_to_unfreeze.append((f"features.denseblock4.{name}", layer, "block4"))
    
    # Get transition3 (between denseblock3 and denseblock4)
    layers_to_unfreeze.append(("features.transition3", model.features.transition3, "transition"))
    
    # Get denseblock3 layers - 24 layers
    denseblock3 = model.features.denseblock3
    for name, layer in reversed(list(denseblock3.named_children())):
        layers_to_unfreeze.append((f"features.denseblock3.{name}", layer, "block3"))
    
    return layers_to_unfreeze


def freeze_all_except_classifier(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_up_to_layer(model, layers_to_unfreeze, num_layers_to_unfreeze):
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Count block4 and block3 layers
    block4_count = sum(1 for _, _, block_type in layers_to_unfreeze if block_type == "block4")
    
    # Determine which layers to unfreeze
    unfrozen_count = 0
    for i, (layer_name, layer, block_type) in enumerate(layers_to_unfreeze):
        if unfrozen_count >= num_layers_to_unfreeze:
            break
            
        # Special handling for transition layer
        if block_type == "transition":
            # Only unfreeze transition if we're starting to unfreeze block3
            # i.e., if we've unfrozen all of block4 (16 layers)
            if num_layers_to_unfreeze > block4_count:
                for param in layer.parameters():
                    param.requires_grad = True
            # Don't count transition in the unfrozen count
            continue
        
        # Unfreeze this layer
        for param in layer.parameters():
            param.requires_grad = True
        unfrozen_count += 1


def create_fresh_model(num_layers_to_unfreeze):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 8)
    
    # Get layers from THIS model, not a reference model
    layers_to_unfreeze = get_densenet_layers_to_unfreeze(model)
    unfreeze_up_to_layer(model, layers_to_unfreeze, num_layers_to_unfreeze)
    
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
        train_folder="~/mcv/datasets/C3/2425/MIT_small_train_1/train",
        test_folder="~/mcv/datasets/C3/2425/MIT_small_train_1/test"
    )

    # Get a reference model to extract layer structure (for printing only)
    reference_model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    layers_info = get_densenet_layers_to_unfreeze(reference_model)
    layer_names = [name for name, _, _ in layers_info]  # Store names only
    del reference_model
    
    print(f"Total layers available to unfreeze: {len(layer_names)}")
    for i, name in enumerate(layer_names):
        print(f"  {i+1}. {name}")

    loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Run independent experiments unfreezing every 4 layers
    # 0 = only classifier, 4 = classifier + 4 layers, 8 = classifier + 8 layers, etc.
    step_size = 4
    # We have 16 (block4) + 1 (transition) + 24 (block3) = 41 total
    # But we only count dense layers (40), so max is 40
    max_layers = sum(1 for _, _, block_type in layers_info if block_type != "transition")
    
    experiments_to_run = [0]  # Start with classifier only
    for num_layers in range(step_size, max_layers + 1, step_size):
        experiments_to_run.append(num_layers)
    # Add the final experiment with all layers if not already included
    if experiments_to_run[-1] != max_layers:
        experiments_to_run.append(max_layers)
    
    num_experiments = len(experiments_to_run)
    print(f"\nWill run {num_experiments} experiments with layer counts: {experiments_to_run}\n")
    
    all_results = []
    
    for exp_idx, num_layers in enumerate(experiments_to_run):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {exp_idx + 1}/{num_experiments}")
        if num_layers == 0:
            print(f"Training: Classifier ONLY")
        else:
            # Determine which layers are unfrozen
            unfrozen_layer_names = []
            transition_unfrozen = False
            count = 0
            for name, _, block_type in layers_info:
                if count >= num_layers:
                    break
                if block_type == "transition":
                    if num_layers > 16:  # Block4 has 16 layers
                        transition_unfrozen = True
                    continue
                unfrozen_layer_names.append(name)
                count += 1
            
            if transition_unfrozen:
                # Insert transition in its proper position (after block4)
                block4_end = next(i for i, name in enumerate(unfrozen_layer_names) 
                                 if not name.startswith("features.denseblock4"))
                unfrozen_layer_names.insert(block4_end, "features.transition3")
            
            print(f"Training: Classifier + {num_layers} layer(s)")
            print(f"Unfrozen layers: {', '.join(unfrozen_layer_names)}")
        print(f"{'='*80}\n")
        
        # Create fresh model for this experiment
        model = create_fresh_model(num_layers)
        
        # Count and display trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
        # Create optimizer for this model
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=1e-5)
        
        # Create experiment name
        if num_layers == 0:
            experiment_name = f"finetune_classifier_only_small_dataset"
        else:
            experiment_name = f"finetune_classifier_plus_{num_layers}_layers_small_dataset"
        
        # Initialize wandb for this experiment
        run = utils.init_wandb_run(
            dry=dry,
            entity="mcv-team-6",
            project="C3-Week3",
            name=f"FT-{num_layers}-layers_small_dataset",
            config={
                "architecture": "DenseNet121-FineTuning",
                "num_unfrozen_layers": num_layers,
                "epochs": EPOCHS,
                "image_size": IMG_SIZE,
                "learning_rate": LR,
                "trainable_params": trainable,
                "total_params": total,
                "trainable_percentage": 100*trainable/total,
            }
        )
        
        # Run experiment
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
        
        # Clean up
        del model
        del optimizer
        torch.cuda.empty_cache()
        
        print(f"\nCompleted experiment {exp_idx + 1}/{num_experiments}\n")
    
    wandb.join()
    
    del train_loader
    del test_loader
    
    print(f"All experiments completed")
    print(f"Total experiments run: {num_experiments}")

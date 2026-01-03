import os
from typing import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.v2  as F
from torchviz import make_dot
import tqdm
import kornia.augmentation as ka

from models.base import WraperModel

import wandb

# Train function
def train(model, dataloader, criterion, optimizer, device, augmentation=None):
    model.train()
    train_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        if augmentation is not None:
            inputs = augmentation(inputs)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        train_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = train_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def test(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = test_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def plot_metrics(train_metrics: Dict, test_metrics: Dict, metric_name: str, directory: str):
    """
    Plots and saves metrics for training and testing.

    Args:
        train_metrics (Dict): Dictionary containing training metrics.
        test_metrics (Dict): Dictionary containing testing metrics.
        metric_name (str): The name of the metric to plot (e.g., "loss", "accuracy").

    Saves:
        - loss.png for loss plots
        - metrics.png for other metrics plots
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics[metric_name], label=f'Train {metric_name.capitalize()}')
    plt.plot(test_metrics[metric_name], label=f'Test {metric_name.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} Over Epochs')
    plt.legend()
    plt.grid(True)

    # Save the plot with the appropriate name
    filename = "loss.png" if metric_name.lower() == "loss" else "metrics.png"
    filename = directory+filename
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

    plt.close()  # Close the figure to free memory

def plot_multi_stage_comparison(all_stages_metrics: List[Dict], metric_name: str, directory: str):
    """
    Plots comparison of metrics across multiple training stages.
    
    Args:
        all_stages_metrics: List of dicts, each containing 'stage_name', 'train_metrics', 'test_metrics'
        metric_name: The metric to plot ('loss' or 'accuracy')
        directory: Directory to save the plot
    """
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.tab10(range(len(all_stages_metrics)))
    
    for idx, stage_data in enumerate(all_stages_metrics):
        stage_name = stage_data['stage_name']
        train_values = stage_data['train_metrics'][metric_name]
        test_values = stage_data['test_metrics'][metric_name]
        
        epochs_offset = sum(len(s['train_metrics'][metric_name]) for s in all_stages_metrics[:idx])
        epochs = range(epochs_offset, epochs_offset + len(train_values))
        
        plt.plot(epochs, train_values, '--', color=colors[idx], alpha=0.6, label=f'{stage_name} (Train)')
        plt.plot(epochs, test_values, '-', color=colors[idx], linewidth=2, label=f'{stage_name} (Test)')
    
    plt.xlabel('Epoch (Cumulative)')
    plt.ylabel(metric_name.capitalize())
    plt.title(f'{metric_name.capitalize()} Across Training Stages')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"multi_stage_{metric_name}.png"
    filepath = directory + filename
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    print(f"Multi-stage comparison plot saved as {filepath}")
    plt.close()

def plot_computational_graph(model: torch.nn.Module, input_size: tuple, filename: str = "computational_graph"):
    """
    Generates and saves a plot of the computational graph of the model.

    Args:
        model (torch.nn.Module): The PyTorch model to visualize.
        input_size (tuple): The size of the dummy input tensor (e.g., (batch_size, input_dim)).
        filename (str): Name of the file to save the graph image.
    """
    model.eval()  # Set the model to evaluation mode
    
    # Generate a dummy input based on the specified input size
    dummy_input = torch.randn(*input_size)

    # Create a graph from the model
    graph = make_dot(model(dummy_input), params=dict(model.named_parameters()), show_attrs=True).render(filename, format="png")

    print(f"Computational graph saved as {filename}")

def make_checkpoint(epoch, model, optimizer, criterion):
    return {
        'epoch': epoch,
        'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
        'optimizer_state_dict': {k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                                for k, v in optimizer.state_dict().items()},
        'loss': criterion,
    }

def experiment(model_folder: str, *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion,
        epochs: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        augmentation: Optional[nn.Module],
        wandb_run: wandb.Run,
        device = None,
        early_stopping_patience: int = 50,
        early_stopping_min_delta: float = 0.0001,
        checkpoint_save_interval: int = 10
    ):
    
    os.makedirs(f"trained_models/{model_folder}", exist_ok=True)
    os.makedirs(f"trained_models/{model_folder}/metrics", exist_ok=True)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    best_test_loss = float('inf')
    best_test_accuracy = 0
    best_test_loss_for_early_stopping = float('inf')
    patience_counter = 0

    best_accuracy_checkpoint = None
    best_loss_checkpoint = None

    for epoch in tqdm.tqdm(range(epochs), desc="TRAINING THE MODEL"):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device, augmentation)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
        
        wandb_run.log({
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_loss": train_loss,
            "test_loss": test_loss,
        })

        is_best_accuracy = best_test_accuracy < test_accuracy and epoch > 10
        is_best_loss = best_test_loss > test_loss and epoch > 10

        if is_best_accuracy or is_best_loss:
            checkpoint = make_checkpoint(epoch, model, optimizer, criterion)

            if is_best_accuracy:
                best_accuracy_checkpoint = checkpoint
                best_test_accuracy = test_accuracy

            if is_best_loss:
                best_loss_checkpoint = checkpoint
                best_test_loss = test_loss
        
        if (epoch + 1) % checkpoint_save_interval == 0:
            if best_accuracy_checkpoint is not None:
                torch.save(best_accuracy_checkpoint, f"trained_models/{model_folder}/best_test_accuracy.pt")
            if best_loss_checkpoint is not None:
                torch.save(best_loss_checkpoint, f"trained_models/{model_folder}/best_test_loss.pt")

        # Early stopping check
        if test_loss < best_test_loss_for_early_stopping - early_stopping_min_delta:
            best_test_loss_for_early_stopping = test_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    print(f"Best test loss: {best_test_loss:.4f}")
    print(f"Best test accuracy: {best_test_accuracy:.4f}")

    # Save to disk at the end
    if best_accuracy_checkpoint is not None:
        torch.save(best_accuracy_checkpoint, f"trained_models/{model_folder}/best_test_accuracy.pt")
    if best_loss_checkpoint is not None:
        torch.save(best_loss_checkpoint, f"trained_models/{model_folder}/best_test_loss.pt")

    # Plot results
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "loss", directory=f"trained_models/{model_folder}/metrics/")
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "accuracy", directory=f"trained_models/{model_folder}/metrics/")

def multi_stage_experiment(
        experiment_folder: str, *,
        stages_config: List[Dict],
        train_loader: DataLoader,
        test_loader: DataLoader,
        augmentation: Optional[nn.Module],
        wandb_run: wandb.Run,
        device = None,
        early_stopping_patience: int = 50,
        early_stopping_min_delta: float = 0.0001,
        checkpoint_save_interval: int = 10
    ):
    
    os.makedirs(f"trained_models/{experiment_folder}", exist_ok=True)
    os.makedirs(f"trained_models/{experiment_folder}/metrics", exist_ok=True)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_stages_metrics = []
    cumulative_epoch = 0
    
    for stage_idx, stage_config in enumerate(stages_config):
        stage_name = stage_config['stage_name']
        model = stage_config['model'].to(device)
        optimizer = stage_config['optimizer']
        criterion = stage_config['criterion']
        epochs = stage_config['epochs']
        
        print(f"\n{'='*60}")
        print(f"Starting {stage_name}")
        print(f"{'='*60}\n")
        
        train_losses, train_accuracies = [], []
        test_losses, test_accuracies = [], []
        
        best_test_loss = float('inf')
        best_test_accuracy = 0
        best_test_loss_for_early_stopping = float('inf')
        patience_counter = 0
        
        best_accuracy_checkpoint = None
        best_loss_checkpoint = None
        
        for epoch in tqdm.tqdm(range(epochs), desc=f"Training {stage_name}"):
            train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device, augmentation)
            test_loss, test_accuracy = test(model, test_loader, criterion, device)
            
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            
            print(f"Epoch {epoch + 1}/{epochs} (Cumulative: {cumulative_epoch + epoch + 1}) - "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            
            wandb_run.log({
                "stage": stage_name,
                "stage_epoch": epoch,
                "cumulative_epoch": cumulative_epoch + epoch,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "train_loss": train_loss,
                "test_loss": test_loss,
            })
            
            is_best_accuracy = best_test_accuracy < test_accuracy and epoch > 10
            is_best_loss = best_test_loss > test_loss and epoch > 10
            
            if is_best_accuracy or is_best_loss:
                checkpoint = make_checkpoint(cumulative_epoch + epoch, model, optimizer, criterion)
                
                if is_best_accuracy:
                    best_accuracy_checkpoint = checkpoint
                    best_test_accuracy = test_accuracy
                
                if is_best_loss:
                    best_loss_checkpoint = checkpoint
                    best_test_loss = test_loss
            
            if (epoch + 1) % checkpoint_save_interval == 0:
                if best_accuracy_checkpoint is not None:
                    torch.save(best_accuracy_checkpoint, f"trained_models/{experiment_folder}/{stage_name}_best_accuracy.pt")
                if best_loss_checkpoint is not None:
                    torch.save(best_loss_checkpoint, f"trained_models/{experiment_folder}/{stage_name}_best_loss.pt")
            
            # Early stopping check
            if test_loss < best_test_loss_for_early_stopping - early_stopping_min_delta:
                best_test_loss_for_early_stopping = test_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch + 1} of {stage_name}")
                break
        
        # Save stage results
        all_stages_metrics.append({
            'stage_name': stage_name,
            'train_metrics': {'loss': train_losses, 'accuracy': train_accuracies},
            'test_metrics': {'loss': test_losses, 'accuracy': test_accuracies},
            'best_test_loss': best_test_loss,
            'best_test_accuracy': best_test_accuracy,
        })
        
        # Save final checkpoints for this stage
        if best_accuracy_checkpoint is not None:
            torch.save(best_accuracy_checkpoint, f"trained_models/{experiment_folder}/{stage_name}_best_accuracy.pt")
        if best_loss_checkpoint is not None:
            torch.save(best_loss_checkpoint, f"trained_models/{experiment_folder}/{stage_name}_best_loss.pt")
        
        # Plot individual stage metrics
        plot_metrics(
            {"loss": train_losses, "accuracy": train_accuracies},
            {"loss": test_losses, "accuracy": test_accuracies},
            "loss",
            directory=f"trained_models/{experiment_folder}/metrics/{stage_name}_"
        )
        plot_metrics(
            {"loss": train_losses, "accuracy": train_accuracies},
            {"loss": test_losses, "accuracy": test_accuracies},
            "accuracy",
            directory=f"trained_models/{experiment_folder}/metrics/{stage_name}_"
        )
        
        cumulative_epoch += len(train_losses)
        
        print(f"\n{stage_name} completed:")
        print(f"  Best test loss: {best_test_loss:.4f}")
        print(f"  Best test accuracy: {best_test_accuracy:.4f}\n")
    
    # Plot multi-stage comparison
    plot_multi_stage_comparison(all_stages_metrics, "loss", directory=f"trained_models/{experiment_folder}/metrics/")
    plot_multi_stage_comparison(all_stages_metrics, "accuracy", directory=f"trained_models/{experiment_folder}/metrics/")
    
    print("Experiment completed")

if __name__ == "__main__":

    FINAL_SIZE = (224, 224)
    FINAL_C = 3

    torch.manual_seed(42)

    train_transformation = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        F.Resize(size=FINAL_SIZE),
    ])
    test_transformation = F.Compose([
        F.ToImage(),
        F.ToDtype(torch.float32, scale=True),
        # F.Resize(size=(128, 128)),
        F.Resize(size=FINAL_SIZE),
    ])
    data_train = ImageFolder("../data/places_reduced/train", transform=train_transformation)
    data_test = ImageFolder("../data/places_reduced/val", transform=test_transformation)

    train_loader = DataLoader(data_train, batch_size=256, pin_memory=True, shuffle=True, num_workers=8, prefetch_factor=4, persistent_workers=True)
    test_loader = DataLoader(data_test, batch_size=128, pin_memory=True, shuffle=False, num_workers=8, prefetch_factor=4, persistent_workers=True)

    C, H, W = np.array(data_train[0][0]).shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # model = SimpleModel(input_d=C*H*W, hidden_d=300, output_d=11)
    # model = make_like_simple(input_d=C*H*W, hidden_d=300, output_d=11)
    model = WraperModel(num_classes=8)
    
    plot_computational_graph(model, input_size=(1, FINAL_C, FINAL_SIZE[0], FINAL_SIZE[1]))  # Batch size of 1, input_dim=10

    model = model.to(device)

    # augmentation = None
    augmentation = nn.Sequential(
        ka.RandomGaussianBlur(kernel_size=(7, 7), sigma=(0.5, 0.5), p=0.1),
        ka.RandomRotation(degrees=(-10, 10)),
        ka.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0), ratio=(1.0, 1.0)),
        ka.ColorJiggle(0.2, 0.2, 0.2, 0.2),
        ka.RandomHorizontalFlip(),
        ka.RandomGrayscale(),
        # # ka.Resize(size=FINAL_SIZE)
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.0001 / 4, weight_decay=1e-6)
    num_epochs = 500

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in tqdm.tqdm(range(num_epochs), desc="TRAINING THE MODEL"):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device, augmentation)
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Plot results
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "loss")
    plot_metrics({"loss": train_losses, "accuracy": train_accuracies}, {"loss": test_losses, "accuracy": test_accuracies}, "accuracy")

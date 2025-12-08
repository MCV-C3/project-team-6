from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import math

def plot_images(images, title="", figsize=(15, 8), max_cols=5):
    """
    Plot a grid of images with their class name and label.
    
    Args:
        images (list): List of (PIL.Image, label) tuples
        title (str): Title for the figure
        figsize (tuple): Figure size
        max_cols (int): Maximum number of columns
    """
    num_imgs = len(images)
    if num_imgs == 0:
        print("No images to plot")
        return
    
    cols = min(num_imgs, max_cols)
    rows = math.ceil(num_imgs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single image case
    if num_imgs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (img, label) in enumerate(images):
        class_name = os.path.basename(os.path.dirname(img.filename))
        filename = os.path.basename(img.filename)
        
        axes[i].imshow(img)
        axes[i].set_title(f"{class_name} ({label})\n{filename}", fontsize=10)
        axes[i].axis("off")
    
    # Hide empty axes if any
    for j in range(num_imgs, len(axes)):
        axes[j].axis("off")
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_cv_accuracy(x_values, means, stds, descriptor_name, hyperparam_name):
    """
    Plot CV validation accuracy vs a hyperparameter using direct arrays.
    """

    plt.figure(figsize=(8, 5))
    plt.errorbar(x_values, means, yerr=stds,
                 marker='o', capsize=5)

    plt.title(f"{descriptor_name} — Validation Accuracy vs {hyperparam_name}\n(5-Fold Cross-Validation)")
    plt.xlabel(hyperparam_name)
    plt.ylabel("Validation Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_cv_accuracy_boxplot(x_values, all_fold_values, descriptor_name, hyperparam_name, ylim=(0, 1), show_points=True):
    """
    Plot CV validation accuracy vs a hyperparameter showing individual fold results.

    For small sample sizes (like 5-fold CV), this shows all individual points
    overlaid on a simple range visualization (min to max with mean).

    Args:
        x_values: List/array of hyperparameter values (e.g., [128, 256, 512])
        all_fold_values: List of lists, where each inner list contains accuracy values
                        from all CV folds for that hyperparameter value
                        Shape: (n_hyperparams, n_folds)
        descriptor_name: Name of the descriptor (e.g., "SIFT")
        hyperparam_name: Name of the hyperparameter (e.g., "Codebook Size")
        ylim: Tuple (ymin, ymax) for y-axis limits. Default is (0, 1).
              Set to None for automatic scaling.
        show_points: If True (default), show individual fold points overlaid on the plot.

    Example:
        x_values = [128, 256, 512]
        all_fold_values = [
            [0.32, 0.33, 0.31, 0.34, 0.32],  # 5 fold accuracies for size=128
            [0.35, 0.36, 0.34, 0.37, 0.35],  # 5 fold accuracies for size=256
            [0.38, 0.39, 0.37, 0.40, 0.38],  # 5 fold accuracies for size=512
        ]
        plot_cv_accuracy_boxplot(x_values, all_fold_values, "SIFT", "Codebook Size")

        # Or with custom y-limits
        plot_cv_accuracy_boxplot(x_values, all_fold_values, "SIFT", "Codebook Size", ylim=(0.2, 0.5))

        # Or with automatic scaling
        plot_cv_accuracy_boxplot(x_values, all_fold_values, "SIFT", "Codebook Size", ylim=None)
    """

    plt.figure(figsize=(10, 6))

    positions = range(len(x_values))

    # For each hyperparameter value, plot min-max range and mean
    for i, (pos, values) in enumerate(zip(positions, all_fold_values)):
        values = np.array(values)
        mean_val = np.mean(values)
        min_val = np.min(values)
        max_val = np.max(values)

        # Plot vertical line from min to max
        plt.plot([pos, pos], [min_val, max_val], 'k-', linewidth=2, alpha=0.3)

        # Plot mean as a horizontal line
        plt.plot([pos - 0.2, pos + 0.2], [mean_val, mean_val], 'r-', linewidth=3, label='Mean' if i == 0 else '')

        # Plot individual points with slight jitter for visibility
        if show_points:
            jitter = np.random.normal(0, 0.05, size=len(values))
            plt.scatter([pos + j for j in jitter], values,
                       color='darkblue', s=80, alpha=0.7, zorder=3,
                       label='Individual Folds' if i == 0 else '')

    # Set x-axis labels
    plt.xticks(positions, x_values)

    # Set y-axis limits
    if ylim is not None:
        plt.ylim(ylim)

    plt.title(f"{descriptor_name} — Validation Accuracy vs {hyperparam_name}\n(5-Fold Cross-Validation)",
              fontsize=14)
    plt.xlabel(hyperparam_name, fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3, axis='y')

    # Add legend
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

def plot_cv_train_test_comparison(x_values, train_fold_values, test_fold_values,
                                   descriptor_name, hyperparam_name, ylim=(0, 1), show_points=True):
    """
    Plot both train and test CV validation accuracy vs a hyperparameter side by side.

    Shows individual fold results for both training and testing performance,
    allowing easy comparison of overfitting/underfitting.

    Args:
        x_values: List/array of hyperparameter values (e.g., [128, 256, 512])
        train_fold_values: List of lists with training accuracy from all CV folds
                          Shape: (n_hyperparams, n_folds)
        test_fold_values: List of lists with test accuracy from all CV folds
                         Shape: (n_hyperparams, n_folds)
        descriptor_name: Name of the descriptor (e.g., "SIFT")
        hyperparam_name: Name of the hyperparameter (e.g., "Codebook Size")
        ylim: Tuple (ymin, ymax) for y-axis limits. Default is (0, 1).
              Set to None for automatic scaling.
        show_points: If True (default), show individual fold points overlaid on the plot.

    Example:
        x_values = [128, 256, 512]
        train_fold_values = [
            [0.45, 0.46, 0.44, 0.47, 0.45],  # 5 fold train accuracies for size=128
            [0.48, 0.49, 0.47, 0.50, 0.48],  # for size=256
            [0.52, 0.53, 0.51, 0.54, 0.52],  # for size=512
        ]
        test_fold_values = [
            [0.32, 0.33, 0.31, 0.34, 0.32],  # 5 fold test accuracies for size=128
            [0.35, 0.36, 0.34, 0.37, 0.35],  # for size=256
            [0.38, 0.39, 0.37, 0.40, 0.38],  # for size=512
        ]
        plot_cv_train_test_comparison(x_values, train_fold_values, test_fold_values,
                                      "SIFT", "Codebook Size")
    """

    plt.figure(figsize=(12, 6))

    positions = np.arange(len(x_values))

    # Plot training results
    for i, (pos, values) in enumerate(zip(positions, train_fold_values)):
        values = np.array(values)
        mean_val = np.mean(values)
        min_val = np.min(values)
        max_val = np.max(values)

        # Plot vertical line from min to max
        plt.plot([pos, pos], [min_val, max_val],
                color='darkgreen', linewidth=3, alpha=0.5, zorder=1)

        # Plot mean as a horizontal line
        plt.plot([pos - 0.15, pos + 0.15], [mean_val, mean_val],
                color='green', linewidth=4, label='Train Mean' if i == 0 else '', zorder=2)

        # Plot individual points with slight jitter for horizontal separation
        if show_points:
            jitter = np.random.normal(-0.08, 0.02, size=len(values))  # Offset left
            plt.scatter([pos + j for j in jitter], values,
                       color='green', s=100, alpha=0.7, zorder=4, marker='o',
                       edgecolors='darkgreen', linewidths=1.5,
                       label='Train Folds' if i == 0 else '')

    # Plot test results (same position, different marker)
    for i, (pos, values) in enumerate(zip(positions, test_fold_values)):
        values = np.array(values)
        mean_val = np.mean(values)
        min_val = np.min(values)
        max_val = np.max(values)

        # Plot vertical line from min to max
        plt.plot([pos, pos], [min_val, max_val],
                color='darkblue', linewidth=3, alpha=0.5, zorder=1)

        # Plot mean as a horizontal line
        plt.plot([pos - 0.15, pos + 0.15], [mean_val, mean_val],
                color='blue', linewidth=4, label='Test Mean' if i == 0 else '', zorder=2)

        # Plot individual points with slight jitter for horizontal separation
        if show_points:
            jitter = np.random.normal(0.08, 0.02, size=len(values))  # Offset right
            plt.scatter([pos + j for j in jitter], values,
                       color='blue', s=100, alpha=0.7, zorder=4, marker='s',
                       edgecolors='darkblue', linewidths=1.5,
                       label='Test Folds' if i == 0 else '')

    # Set x-axis labels
    plt.xticks(positions, x_values)

    # Set y-axis limits
    if ylim is not None:
        plt.ylim(ylim)

    plt.title(f"{descriptor_name} — Train vs Test Accuracy vs {hyperparam_name}\n(5-Fold Cross-Validation)",
              fontsize=14, fontweight='bold')
    plt.xlabel(hyperparam_name, fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3, axis='y')

    # Add legend
    plt.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.show()


def dense_keypoints_grid(img_shape, step, scale):
    h, w = img_shape[:2]
    keypoints = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            keypoints.append(cv2.KeyPoint(x=float(x), y=float(y), size=float(scale)))
    return keypoints

def show_dense_sift(img, step, scale, title=""):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kps = dense_keypoints_grid(gray.shape, step=step, scale=scale)
    img_kps = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.imshow(img_kps)
    plt.axis("off")
    plt.title(f"{title}\nstep={step}, scale={scale}")

def plot_final_metrics_comparison(method_names, metrics_list, title="Final Test Metrics Comparison"):
    """
    Plot a grouped bar chart comparing final metrics for multiple methods.
    
    Args:
        method_names (list[str]): Names of the methods (e.g. ["SIFT (1000 kp)", "Dense SIFT (...)"]).
        metrics_list (list[dict]): List of dicts, one per method, with metric_name -> value.
                                   All dicts must share the same metric keys.
        title (str): Title of the plot.
    """
    # Assume all metric dicts have the same keys and in same order
    metric_names = list(metrics_list[0].keys())
    num_metrics = len(metric_names)
    num_methods = len(method_names)

    # Build matrix of values: shape (num_methods, num_metrics)
    values = np.array([[m[metric] for metric in metric_names] for m in metrics_list])

    x = np.arange(num_metrics)
    width = 0.8 / num_methods  # total width ~0.8

    plt.figure(figsize=(8, 5))
    bars_all = []

    for i in range(num_methods):
        offset = (i - (num_methods - 1) / 2) * width
        bars = plt.bar(x + offset, values[i], width, label=method_names[i])
        bars_all.append(bars)

    # Add value labels on top of bars
    for bars in bars_all:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.xticks(x, metric_names)
    plt.ylabel("Score")
    plt.title(title)
    plt.ylim(0, values.max() + 0.1)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None,
                          title="Confusion Matrix", figsize=(7, 5), cmap="Blues"):
    """
    Plot a confusion matrix with optional class name auto-detection.
    
    Args:
        y_true: Array of true labels
        y_pred: Array of predicted labels
        class_names: Optional list of class names. If None, class indices are used.
        title: Plot title
        figsize: Figure size
        cmap: Colormap
    """
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if class_names is None:
        classes = np.unique(y_true)
        class_names = [f"Class {c}" for c in classes]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=cmap, ax=ax, xticks_rotation=45)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_multiclass_roc(y_true, y_score, class_names=None, figsize=(10, 8)):
    """
    Plot ROC curves for multi-class classification using One-vs-Rest approach.
    Based on scikit-learn's official example.
    
    Args:
        y_true (np.array): True labels (N,). Can be string labels or integers.
        y_score (np.array): Predicted probabilities or decision scores (N, n_classes).
                           Each row should contain the score/probability for each class.
        class_names (list): Optional list of class names for the legend. 
                           If None, uses the unique values from y_true.
        output_path (str): Path where the plot will be saved.
        figsize (tuple): Figure size (width, height).
    
    Returns:
        dict: Dictionary containing fpr, tpr, and roc_auc for each class.
    """
    # Get number of classes
    n_classes = y_score.shape[1]
    
    # Use LabelBinarizer for One-Hot encoding (handles string labels too)
    label_binarizer = LabelBinarizer().fit(y_true)
    y_onehot_test = label_binarizer.transform(y_true)
    
    # Store class names
    if class_names is None:
        class_names = label_binarizer.classes_
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    # Aggregate all false positive rates
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])
    
    # Average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot micro-average ROC curve
    ax.plot(
        fpr["micro"], 
        tpr["micro"],
        label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
        color='deeppink', 
        linestyle=':', 
        linewidth=4
    )
    
    # Plot macro-average ROC curve
    ax.plot(
        fpr["macro"], 
        tpr["macro"],
        label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})',
        color='navy', 
        linestyle=':', 
        linewidth=4
    )
    
    # Plot ROC curve for each class
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 
                    'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
    for class_id, color in zip(range(n_classes), colors):
        ax.plot(
            fpr[class_id], 
            tpr[class_id], 
            color=color, 
            lw=2,
            label=f'ROC curve for {class_names[class_id]} (AUC = {roc_auc[class_id]:.2f})'
        )
    
    # Plot diagonal (chance level)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance level (AUC = 0.5)')
    
    ax.set(
        xlim=[-0.01, 1.01],
        ylim=[-0.01, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="Multi-class ROC curve\nOne-vs-Rest",
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Also print the scores
    print(f"\nMicro-averaged One-vs-Rest ROC AUC score: {roc_auc['micro']:.2f}")
    print(f"Macro-averaged One-vs-Rest ROC AUC score: {roc_auc['macro']:.2f}")
    print("\nPer-class ROC AUC scores:")
    for i in range(n_classes):
        print(f"  {class_names[i]}: {roc_auc[i]:.2f}")

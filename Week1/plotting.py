import matplotlib.pyplot as plt
import cv2
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import os
import math

def plot_one_sample_per_class(dataset, title="One Example Image per Class"):
    """
    Plots one sample image per class from the dataset in 2 rows.
    
    Args:
        dataset: List of (PIL.Image, label)
        title: Optional title for the figure
    """

    # Extract class names and create mapping class_name -> label
    class_dirs = {
        os.path.basename(os.path.dirname(img.filename)): label
        for img, label in dataset
    }
    classes = sorted(class_dirs.keys(), key=lambda c: class_dirs[c])

    num_classes = len(classes)
    rows = 2
    cols = math.ceil(num_classes / rows)

    # Collect one example image per class
    examples = {}
    for img, label in dataset:
        cls_name = os.path.basename(os.path.dirname(img.filename))
        if cls_name not in examples:
            examples[cls_name] = (img, label)
        if len(examples) == num_classes:
            break

    # Plot
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    for idx, cls_name in enumerate(classes):
        img, label = examples[cls_name]
        axes[idx].imshow(np.array(img))
        axes[idx].set_title(f"{cls_name} ({label})")
        axes[idx].axis("off")

    # Hide empty axes if any
    for j in range(len(classes), len(axes)):
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

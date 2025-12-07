import matplotlib.pyplot as plt
import cv2

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

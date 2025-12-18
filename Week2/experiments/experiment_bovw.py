import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import utils

from models.patch_based_classifier import make_patch_model
from pipeline import test

argparser = utils.get_experiment_argument_parser()
argparser.add_argument('--checkpoint-path', type=str, required=True, help='Path to the saved model checkpoint')
argparser.add_argument('--patch-size', type=int, default=32, help='Size of the square patches')
argparser.add_argument('--stride', type=int, default=None, help='Stride for patch extraction (default: same as patch_size for non-overlapping)')
argparser.add_argument('--depth', type=int, default=2, help='Depth of the descriptor MLP')
argparser.add_argument('--last-layer-width', type=int, default=512, help='Width of the last hidden layer in descriptor MLP')
argparser.add_argument('--merge-strategy', type=str, default='mean', help='Merge strategy (used only for model architecture, not for BoVW)')
argparser.add_argument('--num-words', type=int, default=512, help='Number of visual words (KMeans clusters)')
argparser.add_argument('--lda-components', type=int, default=None, help='Number of LDA components for dimensionality reduction (default: None, no LDA)')
argparser.add_argument('--imsize', type=int, default=224, help='Side length of the image')
args = argparser.parse_args()

CHECKPOINT_PATH = args.checkpoint_path
PATCH_SIZE = args.patch_size
STRIDE = args.stride if args.stride is not None else PATCH_SIZE
DEPTH = args.depth
LAST_LAYER_WIDTH = args.last_layer_width
MERGE_STRATEGY = args.merge_strategy
NUM_WORDS = args.num_words
LDA_COMPONENTS = args.lda_components
IMG_SIZE = args.imsize

device = utils.set_device(args.gpu_id)

train_loader, test_loader = utils.get_loaders(image_size=(IMG_SIZE, IMG_SIZE), resize_train=True)

# Build descriptor widths: all layers are 512 except the last one
descriptor_widths = [512] * (DEPTH - 1) + [LAST_LAYER_WIDTH]

model = make_patch_model(
    input_channels=3,
    patch_size=PATCH_SIZE,
    stride=STRIDE,
    descriptor_widths=descriptor_widths,
    num_classes=11,
    merge_strategy=MERGE_STRATEGY,
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"Loaded model from {CHECKPOINT_PATH}")
print(f"Model architecture: patch_size={PATCH_SIZE}, stride={STRIDE}, depth={DEPTH}, last_layer_width={LAST_LAYER_WIDTH}")
print(f"Bag of Visual Words: num_words={NUM_WORDS}")
if LDA_COMPONENTS is not None:
    print(f"LDA dimensionality reduction: {LDA_COMPONENTS} components")

# Sanity check: Evaluate the network directly on test set
print("\n" + "="*60)
print("SANITY CHECK: Evaluating loaded model directly on test set...")
print("="*60)
loss = torch.nn.CrossEntropyLoss()
test_loss, test_accuracy = test(model, test_loader, loss, device)
print(f"Direct model test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Direct model test loss: {test_loss:.4f}")
print("="*60)

# Step 2: Generate descriptors for training images
print("\n" + "="*60)
print("Step 1/5: Extracting patch descriptors for training set...")
print("="*60)
train_descriptors_list = []
train_labels = []

with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        # Get descriptors for all patches (returns [batch_size * num_patches, descriptor_dim])
        descriptors = model.get_descriptors(images)
        train_descriptors_list.append(descriptors.cpu().numpy())
        train_labels.append(labels.cpu().numpy())

# All patch descriptors from all training images
all_train_descriptors = np.vstack(train_descriptors_list)
train_labels = np.concatenate(train_labels)

print(f"Total patch descriptors extracted: {all_train_descriptors.shape}")
print(f"Train labels shape: {train_labels.shape}")

# Calculate number of patches per image
num_patches_h = (IMG_SIZE - PATCH_SIZE) // STRIDE + 1
num_patches_w = (IMG_SIZE - PATCH_SIZE) // STRIDE + 1
num_patches_per_image = num_patches_h * num_patches_w

print(f"Number of patches per image: {num_patches_per_image}")

# Optional LDA dimensionality reduction (must be done BEFORE KMeans)
lda = None
if LDA_COMPONENTS is not None:
    print("\n" + "="*60)
    print(f"Step 2/5: Applying LDA dimensionality reduction to {LDA_COMPONENTS} components...")
    print("="*60)

    # Create labels by repeating each label num_patches_per_image times
    descriptor_labels = np.repeat(train_labels, num_patches_per_image)

    print(f"Descriptor labels shape: {len(descriptor_labels)}")

    lda = LinearDiscriminantAnalysis(n_components=LDA_COMPONENTS)
    all_train_descriptors = lda.fit_transform(all_train_descriptors, descriptor_labels)
    print(f"LDA completed! Descriptors reduced to shape: {all_train_descriptors.shape}")

# Step 3: Train KMeans on the descriptors (after LDA if applicable)
print("\n" + "="*60)
if LDA_COMPONENTS is not None:
    print("Step 3/5: Training MiniBatchKMeans for visual vocabulary...")
else:
    print("Step 2/5: Training MiniBatchKMeans for visual vocabulary...")
print("="*60)

# Set batch_size as 10% of the number of descriptors or 10 * num_words, whichever is larger
# This ensures reasonable memory usage while maintaining statistical quality
batch_size = max(NUM_WORDS * 10, int(all_train_descriptors.shape[0] * 0.1))
print(f"Using batch_size={batch_size} for MiniBatchKMeans")

kmeans = MiniBatchKMeans(
    n_clusters=NUM_WORDS,
    batch_size=batch_size,
    random_state=42,
    verbose=0
)

kmeans.fit(all_train_descriptors)
print(f"KMeans training completed! Vocabulary of {NUM_WORDS} visual words created.")

# Step 4 & 5: For each image, compute word assignments and create histograms
print("\n" + "="*60)
if LDA_COMPONENTS is not None:
    print("Step 4/5: Computing BoVW histograms for training set...")
else:
    print("Step 3/5: Computing BoVW histograms for training set...")
print("="*60)

# Compute histograms for training set
train_histograms = []
num_train_images = len(train_labels)

for i in range(num_train_images):
    # Get descriptors for this image
    start_idx = i * num_patches_per_image
    end_idx = start_idx + num_patches_per_image
    image_descriptors = all_train_descriptors[start_idx:end_idx]

    # Assign each descriptor to a word
    words = kmeans.predict(image_descriptors)

    # Create normalized histogram (L1 normalization)
    histogram = np.bincount(words, minlength=NUM_WORDS).astype(float)
    histogram /= histogram.sum()  # L1 normalization (sum to 1)

    train_histograms.append(histogram)

train_histograms = np.array(train_histograms)
print(f"Train histograms shape: {train_histograms.shape}")

# Free up memory
del all_train_descriptors
del train_descriptors_list

# Step 6: Train SVC
print("\n" + "="*60)
if LDA_COMPONENTS is not None:
    print("Step 5/5: Training SVC with RBF kernel, C=1...")
else:
    print("Step 4/5: Training SVC with RBF kernel, C=1...")
print("="*60)
svc = SVC(kernel='rbf', C=1.0, verbose=False)
svc.fit(train_histograms, train_labels)
print("SVC training completed!")

# Test set processing
print("\n" + "="*60)
print("Evaluating: Computing BoVW histograms for test set...")
print("="*60)
test_descriptors_list = []
test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        descriptors = model.get_descriptors(images)
        test_descriptors_list.append(descriptors.cpu().numpy())
        test_labels.append(labels.cpu().numpy())

all_test_descriptors = np.vstack(test_descriptors_list)
test_labels = np.concatenate(test_labels)

print(f"Total test patch descriptors extracted: {all_test_descriptors.shape}")
print(f"Test labels shape: {test_labels.shape}")

# Apply LDA transformation if it was used on training data
if lda is not None:
    print(f"Applying LDA transformation to test descriptors...")
    all_test_descriptors = lda.transform(all_test_descriptors)
    print(f"Test descriptors after LDA: {all_test_descriptors.shape}")

# Compute histograms for test set
test_histograms = []
num_test_images = len(test_labels)

for i in range(num_test_images):
    start_idx = i * num_patches_per_image
    end_idx = start_idx + num_patches_per_image
    image_descriptors = all_test_descriptors[start_idx:end_idx]

    words = kmeans.predict(image_descriptors)

    histogram = np.bincount(words, minlength=NUM_WORDS).astype(float)
    histogram /= histogram.sum()  # L1 normalization

    test_histograms.append(histogram)

test_histograms = np.array(test_histograms)
print(f"Test histograms shape: {test_histograms.shape}")

# Evaluate
print("\n" + "="*60)
print("Evaluating SVC on test set...")
print("="*60)
test_predictions = svc.predict(test_histograms)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"\n{'='*60}")
print(f"RESULTS:")
print(f"  Checkpoint: {CHECKPOINT_PATH}")
print(f"  Patch size: {PATCH_SIZE}x{PATCH_SIZE}, stride: {STRIDE}")
print(f"  Descriptor depth: {DEPTH}, last layer width: {LAST_LAYER_WIDTH}")
print(f"  Number of visual words: {NUM_WORDS}")
if LDA_COMPONENTS is not None:
    print(f"  LDA components: {LDA_COMPONENTS}")
print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"{'='*60}")

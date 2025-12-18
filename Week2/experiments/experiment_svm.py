import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import utils
from models.descriptor_classifier import make_from_widths

argparser = utils.get_experiment_argument_parser()
argparser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the saved model checkpoint')
argparser.add_argument('--depth', type=int, default=2, help='Depth of the hidden layers')
argparser.add_argument('--width', type=int, default=300, help='Width of the hidden layers')
argparser.add_argument('--imsize', type=int, default=224, help='Side length of the image')
args = argparser.parse_args()

CHECKPOINT_PATH = args.checkpoint_path
SIDE = args.imsize
WIDTH = args.width
DEPTH = args.depth

device = utils.set_device(args.gpu_id)

train_loader, test_loader = utils.get_loaders(image_size=(SIDE, SIDE), resize_train=True, resize_test=True)

model = make_from_widths(3*SIDE*SIDE, [WIDTH]*DEPTH, [11])

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"Loaded model from {CHECKPOINT_PATH}")
print(f"Model architecture: depth={DEPTH}, width={WIDTH}, image_size={SIDE}x{SIDE}")

print("Extracting descriptors for training set...")
train_descriptors = []
train_labels = []

with torch.no_grad():
    for images, labels in train_loader:
        images = images.to(device)
        descriptors = model.get_descriptors(images)
        train_descriptors.append(descriptors.cpu().numpy())
        train_labels.append(labels.cpu().numpy())

train_descriptors = np.vstack(train_descriptors)
train_labels = np.concatenate(train_labels)

print(f"Train descriptors shape: {train_descriptors.shape}")
print(f"Train labels shape: {train_labels.shape}")

print("Training SVC with RBF kernel, C=1...")
svc = SVC(kernel='rbf', C=1.0)
svc.fit(train_descriptors, train_labels)
print("SVC training completed!")

print("Extracting descriptors for test set...")
test_descriptors = []
test_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        descriptors = model.get_descriptors(images)
        test_descriptors.append(descriptors.cpu().numpy())
        test_labels.append(labels.cpu().numpy())

test_descriptors = np.vstack(test_descriptors)
test_labels = np.concatenate(test_labels)

print(f"Test descriptors shape: {test_descriptors.shape}")
print(f"Test labels shape: {test_labels.shape}")

print("Evaluating SVC on test set...")
test_predictions = svc.predict(test_descriptors)
test_accuracy = accuracy_score(test_labels, test_predictions)

print(f"\n{'='*50}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"{'='*50}")

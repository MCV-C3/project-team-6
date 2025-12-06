import shutil
import statistics
import cv2
from sklearn.svm import SVC
from bovw import BOVW

from typing import *
from PIL import Image

import numpy as np
import glob
import tqdm
import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sklearn
from sklearn.model_selection import StratifiedKFold

def get_detector_config_id(bovw: BOVW) -> str:
    """
    Generates a unique ID string based on detector type and its parameters.
    """
    cfg = bovw.detector_type

    if hasattr(bovw, "detector_kwargs") and len(bovw.detector_kwargs) > 0:
        for key, value in sorted(bovw.detector_kwargs.items()):
            cfg += f"_{key}{value}"
            
    if hasattr(bovw, "dense_kwargs") and len(bovw.dense_kwargs) > 0:
        for key, value in sorted(bovw.dense_kwargs.items()):
            cfg += f"_{key}{value}"

    return cfg


_CACHE_ENABLED = True
_CACHE_FOLDER = "cache_descriptors"

def enable_cache():
    global _CACHE_ENABLED
    _CACHE_ENABLED = True
    

def disable_cache():
    global _CACHE_ENABLED
    _CACHE_ENABLED = False


def cache_is_enabled():
    global _CACHE_ENABLED
    return _CACHE_ENABLED


def clear_cache():
    shutil.rmtree(_CACHE_FOLDER, ignore_errors=True)


def setup_cache(bovw: BOVW, type: str) -> str:
    # Build cache directory
    config_id = get_detector_config_id(bovw)
    cache_dir = os.path.join(_CACHE_FOLDER, config_id, type)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_image_cache_path(cache_dir: str, image: Image) -> str:
    # Define cache path based on image filename
    img_path = getattr(image, "filename", None)
    if img_path is not None:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        cache_path = os.path.join(cache_dir, base_name + ".pkl")
        return cache_path

    raise ValueError("img_path was None.")


def serialize_keypoints(keypoints: list[cv2.KeyPoint]) -> list[dict]:
    return [
        {
            "pt": kp.pt,
            "size": kp.size,
            "angle": kp.angle,
            "response": kp.response,
            "octave": kp.octave,
            "class_id": kp.class_id
        }
        for kp in keypoints
    ]


def deserialize_keypoints(keypoints_data: list[dict]) -> list[cv2.KeyPoint]:
    return [
        cv2.KeyPoint(
            x=kp["pt"][0],
            y=kp["pt"][1],
            size=kp["size"],
            angle=kp["angle"],
            response=kp["response"],
            octave=kp["octave"],
            class_id=kp["class_id"]
        )
        for kp in keypoints_data
    ]


def load_cached_descriptors(cache_path: str) -> Tuple[Optional[np.ndarray], Optional[list[cv2.KeyPoint]]]:
    
    if not _CACHE_ENABLED:
        return None, None
    
    # If pickle exists, load descriptors from cache
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            return data["descriptors"], deserialize_keypoints(data["keypoints"])

    return None, None


def cache_descriptors(cache_path: str, descriptors: np.ndarray, keypoints: list[cv2.KeyPoint], label):
    
    if not _CACHE_ENABLED:
        return
    
    with open(cache_path, "wb") as f:
        serializable_keypoints = serialize_keypoints(keypoints)
        pickle.dump({"descriptors": descriptors, "keypoints": serializable_keypoints, "label": label}, f)


def extract_bovw_histograms(bovw: BOVW, descriptors: Literal["N", "T", "d"], keypoints: list[list[cv2.KeyPoint]], image_sizes: list[Tuple[int, int]]):
    return np.array([bovw._compute_codebook_descriptor(descriptors=descriptor, keypoints=keypoint, kmeans=bovw.codebook_algo, image_size=image_size) for descriptor, keypoint, image_size in zip(descriptors, keypoints, image_sizes)])


def test(dataset: List[Tuple[Type[Image.Image], int]],
        bovw: BOVW, 
        classifier:Type[object],
    ):
    
    test_descriptors = []
    test_keypoints = []
    descriptors_labels = []
    test_image_resolutions = []

    cache_dir = setup_cache(bovw, "val")
    
    for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Eval]: Extracting the descriptors"):
        image, label = dataset[idx]
        
        cache_path = get_image_cache_path(cache_dir, image)
        descriptors, keypoints = load_cached_descriptors(cache_path)
        if descriptors is None or keypoints is None:
            keypoints, descriptors = bovw._extract_features(image=np.array(image))
            cache_descriptors(cache_path, descriptors, keypoints, label)
        
        if descriptors is not None:
            descriptors = bovw.normalize_descriptors(descriptors)
            test_descriptors.append(descriptors)
            test_keypoints.append(keypoints)
            descriptors_labels.append(label)
            test_image_resolutions.append((image.height, image.width))
    
    test_descriptors = bovw.reduce_dimensionality(test_descriptors)
    
    test_descriptors = bovw.scale_all_descriptors(test_descriptors)
    
    print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms(descriptors=test_descriptors, keypoints=test_keypoints, image_sizes=test_image_resolutions, bovw=bovw)
    
    print("predicting the values")
    y_pred = classifier.predict(bovw_histograms)
    
    acc = accuracy_score(y_true=descriptors_labels, y_pred=y_pred)
    prec = precision_score(y_true=descriptors_labels, y_pred=y_pred, average='weighted')
    rec = recall_score(y_true=descriptors_labels, y_pred=y_pred, average='weighted')
    f1 = f1_score(y_true=descriptors_labels, y_pred=y_pred, average='weighted')

    print("Accuracy on Phase[Test]:", acc)
    print("Precision on Phase[Test]:", prec)
    print("Recall on Phase[Test]:", rec)
    print("F1-Score on Phase[Test]:", f1)

    return acc, prec, rec, f1
    

def train(dataset: List[Tuple[Type[Image.Image], int]], bovw:BOVW, classifier: sklearn.base.BaseEstimator):
    all_descriptors = []
    all_keypoints = []
    all_labels = []
    all_image_resolutions = []

    cache_dir = setup_cache(bovw, "train")
    
    for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Training]: Extracting the descriptors"):
        
        image, label = dataset[idx]

        cache_path = get_image_cache_path(cache_dir, image)
        descriptors, keypoints = load_cached_descriptors(cache_path)
        if descriptors is None or keypoints is None:
            keypoints, descriptors = bovw._extract_features(image=np.array(image))
            cache_descriptors(cache_path, descriptors, keypoints, label)
        
        if descriptors is not None:
            descriptors = bovw.normalize_descriptors(descriptors)
            all_descriptors.append(descriptors)
            all_keypoints.append(keypoints)
            all_labels.append(label)
            all_image_resolutions.append((image.height, image.width))
    
    all_descriptors = bovw.fit_reduce_dimensionality(all_descriptors, all_labels)
    
    all_descriptors = bovw.fit_scale_descriptors_jointly(all_descriptors)
    
    print("Fitting the codebook")
    kmeans, cluster_centers = bovw._update_fit_codebook(descriptors=all_descriptors)

    print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms(descriptors=all_descriptors, keypoints=all_keypoints, image_sizes=all_image_resolutions, bovw=bovw) 
    
    print("Fitting the classifier")
    classifier = classifier.fit(bovw_histograms, all_labels)

    y_pred = classifier.predict(bovw_histograms)
    print("Accuracy on Phase[Train]:", accuracy_score(y_true=all_labels, y_pred=y_pred))
    print("Precision on Phase[Train]:", precision_score(y_true=all_labels, y_pred=y_pred, average='weighted'))
    print("Recall on Phase[Train]:", recall_score(y_true=all_labels, y_pred=y_pred, average='weighted'))
    print("F1-Score on Phase[Train]:", f1_score(y_true=all_labels, y_pred=y_pred, average='weighted'))
    
    return bovw, classifier

class Score(NamedTuple):
    mean: float
    std: float
    all: list[float]

def scores_stats(scores: list[float]) -> Score:
    mean = statistics.mean(scores)
    std = statistics.stdev(scores)
    return Score(mean, std, scores)

class Scores(NamedTuple):
    accuracy: Score
    precision: Score
    recall: Score
    f1: Score
    

class CVResult(NamedTuple):
    train: Scores
    test: Scores

class FullEntry(NamedTuple):
    image: Image.Image
    descriptors: np.ndarray
    keypoints: list[cv2.KeyPoint]
    resolution: tuple[int, int]
    label: int

def compute_descriptors_and_keypoints_for_cv(dataset: List[Tuple[Type[Image.Image], int]], bovw: BOVW) -> list[FullEntry]:
    final_dataset = []
    
    cache_dir = setup_cache(bovw, "train")

    for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Setup]: Extracting the descriptors"):
        
        image, label = dataset[idx]

        cache_path = get_image_cache_path(cache_dir, image)
        descriptors, keypoints = load_cached_descriptors(cache_path)
        if descriptors is None or keypoints is None:
            keypoints, descriptors = bovw._extract_features(image=np.array(image))
            cache_descriptors(cache_path, descriptors, keypoints, label)
        
        if descriptors is None:
            print(f"Could not compute descriptors for image {image.filename} of class {label}.")
            # FIXME: do something about this? Maybe np.empty descriptors and [] keypoints? For the moment we just skip.
            continue
        
        
        descriptors = bovw.normalize_descriptors(descriptors) # FIXME: here is ok? Maybe this will modify BOVW in the future.
        resolution = (image.height, image.width)
        
        final_dataset.append(FullEntry(image, descriptors, keypoints, resolution, label))
        
    return final_dataset


type CVDataset = List[FullEntry]

def test_for_cv(test_data: CVDataset, bovw:BOVW, classifier: sklearn.base.BaseEstimator, verbose: bool = True):
    test_descriptors = [entry.descriptors for entry in test_data]
    test_keypoints = [entry.keypoints for entry in test_data]
    test_image_resolutions = [entry.resolution for entry in test_data]
    test_labels = [entry.label for entry in test_data]

    test_descriptors = bovw.reduce_dimensionality(test_descriptors)
    
    test_descriptors = bovw.scale_all_descriptors(test_descriptors)

    if verbose:
        print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms(descriptors=test_descriptors, keypoints=test_keypoints, image_sizes=test_image_resolutions, bovw=bovw)

    if verbose:
        print("predicting the values")
    y_pred = classifier.predict(bovw_histograms)
    
    acc = accuracy_score(y_true=test_labels, y_pred=y_pred)
    prec = precision_score(y_true=test_labels, y_pred=y_pred, average='weighted')
    rec = recall_score(y_true=test_labels, y_pred=y_pred, average='weighted')
    f1 = f1_score(y_true=test_labels, y_pred=y_pred, average='weighted')

    # print("Accuracy on Phase[Test]:", acc)
    # print("Precision on Phase[Test]:", prec)
    # print("Recall on Phase[Test]:", rec)
    # print("F1-Score on Phase[Test]:", f1)

    return acc, prec, rec, f1

def train_for_cv(train_data: CVDataset, bovw:BOVW, classifier: sklearn.base.BaseEstimator, verbose: bool = True):
    all_descriptors = [entry.descriptors for entry in train_data]
    all_keypoints = [entry.keypoints for entry in train_data]
    all_image_resolutions = [entry.resolution for entry in train_data]
    all_labels = [entry.label for entry in train_data]
    
    all_descriptors = bovw.fit_reduce_dimensionality(all_descriptors, all_labels)

    all_descriptors = bovw.fit_scale_descriptors_jointly(all_descriptors)

    if verbose:
        print("Fitting the codebook")
    kmeans, cluster_centers = bovw._update_fit_codebook(descriptors=all_descriptors)

    if verbose:
        print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms(descriptors=all_descriptors, keypoints=all_keypoints, image_sizes=all_image_resolutions, bovw=bovw)

    if verbose:
        print("Fitting the classifier")
    classifier = classifier.fit(bovw_histograms, all_labels)

    y_pred = classifier.predict(bovw_histograms)

    accuracy = accuracy_score(y_true=all_labels, y_pred=y_pred)
    precision = precision_score(y_true=all_labels, y_pred=y_pred, average='weighted')
    recall = recall_score(y_true=all_labels, y_pred=y_pred, average='weighted')
    f1 = f1_score(y_true=all_labels, y_pred=y_pred, average='weighted')

    if verbose:
        print("Accuracy on Phase[Train]:", accuracy)
        print("Precision on Phase[Train]:", precision)
        print("Recall on Phase[Train]:", recall)
        print("F1-Score on Phase[Train]:", f1)
    
    scores = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
    return bovw, classifier, scores

def cross_validate_bovw(dataset, bovw_kwargs, classifier_cls, classifier_kwargs, n_splits=5, verbose: bool = False) -> CVResult:

    # descriptor generation can be done in common, since NOTHING is learnt
    bovw = BOVW(**bovw_kwargs)
    full_dataset = compute_descriptors_and_keypoints_for_cv(dataset, bovw)

    images = [entry.image for entry in full_dataset]
    labels = np.array([entry.label for entry in full_dataset])

    if verbose:
        print(f"Starting {n_splits}-Fold Cross-Validation")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1s = []

    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_f1s = []

    for fold_idx, (train_idx, val_idx) in tqdm.tqdm(enumerate(skf.split(images, labels), 1), total=n_splits):
        if verbose:
            print(f"\n[FOLD {fold_idx}/{n_splits}]")
            print(f"  Training Samples: {len(train_idx)} | Validation Samples: {len(val_idx)}")

        train_data = [full_dataset[i] for i in train_idx]
        val_data   = [full_dataset[i] for i in val_idx]

        bovw = BOVW(**bovw_kwargs)
        classifier = classifier_cls(**classifier_kwargs)

        bovw, classifier, train_scores = train_for_cv(train_data, bovw, classifier, verbose=verbose)
        acc, prec, rec, f1 = test_for_cv(val_data, bovw, classifier, verbose=verbose)

        train_accuracies.append(train_scores["accuracy"])
        train_precisions.append(train_scores["precision"])
        train_recalls.append(train_scores["recall"])
        train_f1s.append(train_scores["f1"])

        test_accuracies.append(acc)
        test_precisions.append(prec)
        test_recalls.append(rec)
        test_f1s.append(f1)

    # print(f"\n{n_splits}-Fold Cross-Validation Results")
    # print(f"Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    # print(f"Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
    # print(f"Recall: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
    # print(f"F1-score: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    test_scores = Scores(
        accuracy=scores_stats(test_accuracies),
        precision=scores_stats(test_precisions),
        recall=scores_stats(test_recalls),
        f1=scores_stats(test_f1s),
    )
    
    train_scores = Scores(
        accuracy=scores_stats(train_accuracies),
        precision=scores_stats(train_precisions),
        recall=scores_stats(train_recalls),
        f1=scores_stats(train_f1s),
    )

    return CVResult(
        train=train_scores,
        test=test_scores,
    )



def Dataset(ImageFolder:str = "../data/places_reduced") -> List[Tuple[Type[Image.Image], int]]:

    """
    Expected Structure:

        ImageFolder/<cls label>/xxx1.png
        ImageFolder/<cls label>/xxx2.png
        ImageFolder/<cls label>/xxx3.png
        ...

        Example:
            ImageFolder/cat/123.png
            ImageFolder/cat/nsdf3.png
            ImageFolder/cat/[...]/asd932_.png
    
    """

    class_folders = list(sorted(os.listdir(ImageFolder)))

    map_classes = {clsi: idx for idx, clsi  in enumerate(class_folders)}
    
    dataset :List[Tuple] = []

    for idx, cls_folder in enumerate(class_folders):

        image_path = os.path.join(ImageFolder, cls_folder)
        images: List[str] = glob.glob(image_path+"/*.jpg")
        for img in images:
            img_pil = Image.open(img).convert("RGB")
            img_pil.filename = img 
            dataset.append((img_pil, map_classes[cls_folder]))


    return dataset





if __name__ == "__main__":
     #../data/places_reduced/
    data_train = Dataset(ImageFolder="../data/places_reduced/train")
    data_test = Dataset(ImageFolder="../data/places_reduced/val") 

    bovw = BOVW(detector_type="SIFT", descriptor_normalization="L1", codebook_size=5000, dense_kwargs={"step": 32})
    bovw = BOVW(codebook_size=200, pyramid_levels=3) # default works the same
    classifier = LogisticRegression(class_weight="balanced")
    classifier = SVC(kernel='rbf')
    
    bovw, classifier = train(dataset=data_train, bovw=bovw, classifier=classifier)
    
    test(dataset=data_test, bovw=bovw, classifier=classifier)
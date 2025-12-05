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


def get_detector_config_id(bovw: Type[BOVW]) -> str:
    """
    Generates a unique ID string based on detector type and its parameters.
    """
    cfg = bovw.detector_type

    if hasattr(bovw, "detector_kwargs") and len(bovw.detector_kwargs) > 0:
        for key, value in sorted(bovw.detector_kwargs.items()):
            cfg += f"_{key}{value}"
            
    if hasattr(bovw, "dense_kwargs") and len(bovw.detector_kwargs) > 0:
        for key, value in sorted(bovw.dense_kwargs.items()):
            cfg += f"_{key}{value}"

    return cfg

def setup_cache(bovw: BOVW, type: str) -> str:
    # Build cache directory
    config_id = get_detector_config_id(bovw)
    cache_dir = os.path.join("cache_descriptors", config_id, type)
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
    # If pickle exists, load descriptors from cache
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            return data["descriptors"], data["keypoints"]

    return None, None


def cache_descriptors(cache_path: str, descriptors: np.ndarray, keypoints: list[cv2.KeyPoint], label):
    
    return # TODO: remove this to allow caching (my drive is almost full)
    
    with open(cache_path, "wb") as f:
        serializable_keypoints = serialize_keypoints(keypoints)
        pickle.dump({"descriptors": descriptors, "keypoints": serializable_keypoints, "label": label}, f)


def extract_bovw_histograms(bovw: Type[BOVW], descriptors: Literal["N", "T", "d"], keypoints: list[list[cv2.KeyPoint]], image_sizes: list[Tuple[int, int]]):
    return np.array([bovw._compute_codebook_descriptor(descriptors=descriptor, keypoints=keypoint, kmeans=bovw.codebook_algo, image_size=image_size) for descriptor, keypoint, image_size in zip(descriptors, keypoints, image_sizes)])


def test(dataset: List[Tuple[Type[Image.Image], int]],
        bovw: Type[BOVW], 
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
    
    print("Accuracy on Phase[Test]:", accuracy_score(y_true=descriptors_labels, y_pred=y_pred))
    print("Precision on Phase[Test]:", precision_score(y_true=descriptors_labels, y_pred=y_pred, average='weighted'))
    print("Recall on Phase[Test]:", recall_score(y_true=descriptors_labels, y_pred=y_pred, average='weighted'))
    print("F1-Score on Phase[Test]:", f1_score(y_true=descriptors_labels, y_pred=y_pred, average='weighted'))
    

def train(dataset: List[Tuple[Type[Image.Image], int]], bovw:Type[BOVW], classifier: sklearn.base.BaseEstimator):
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
    
    all_descriptors = bovw.fit_reduce_dimensionality(all_descriptors)
    
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
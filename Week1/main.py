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
from sklearn.metrics import accuracy_score
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


def load_cached_descriptors(cache_path: str) -> Optional[np.ndarray]:
    # If pickle exists, load descriptors from cache
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            return data["descriptors"]
    
    return None


def cache_descriptors(cache_path: str, descriptors: np.ndarray, label):
    with open(cache_path, "wb") as f:
        pickle.dump({"descriptors": descriptors, "label": label}, f)


def extract_bovw_histograms(bovw: Type[BOVW], descriptors: Literal["N", "T", "d"]):
    return np.array([bovw._compute_codebook_descriptor(descriptors=descriptor, kmeans=bovw.codebook_algo) for descriptor in descriptors])


def test(dataset: List[Tuple[Type[Image.Image], int]],
        bovw: Type[BOVW], 
        classifier:Type[object],
        scaler: Optional[Any]
    ):
    
    test_descriptors = []
    descriptors_labels = []

    cache_dir = setup_cache(bovw, "val")
    
    for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Eval]: Extracting the descriptors"):
        image, label = dataset[idx]
        
        cache_path = get_image_cache_path(cache_dir, image)
        descriptors = load_cached_descriptors(cache_path)
        if descriptors is None:
            _, descriptors = bovw._extract_features(image=np.array(image))
            cache_descriptors(cache_path, descriptors, label)
        
        if descriptors is not None:
            descriptors = bovw.normalize_descriptors(descriptors)
            test_descriptors.append(descriptors)
            descriptors_labels.append(label)
    
    if scaler:
        test_descriptors = [scaler.transform(descriptors) for descriptors in test_descriptors]
    
    print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms(descriptors=test_descriptors, bovw=bovw)
    
    print("predicting the values")
    y_pred = classifier.predict(bovw_histograms)
    
    print("Accuracy on Phase[Test]:", accuracy_score(y_true=descriptors_labels, y_pred=y_pred))
    

def train(dataset: List[Tuple[Type[Image.Image], int]], bovw:Type[BOVW], classifier: sklearn.base.BaseEstimator):
    all_descriptors = []
    all_labels = []

    cache_dir = setup_cache(bovw, "train")
    
    for idx in tqdm.tqdm(range(len(dataset)), desc="Phase [Training]: Extracting the descriptors"):
        
        image, label = dataset[idx]

        cache_path = get_image_cache_path(cache_dir, image)
        descriptors = load_cached_descriptors(cache_path)
        if descriptors is None:
            _, descriptors = bovw._extract_features(image=np.array(image))
            cache_descriptors(cache_path, descriptors, label)
        
        if descriptors is not None:
            descriptors = bovw.normalize_descriptors(descriptors)
            all_descriptors.append(descriptors)
            all_labels.append(label)
    
    all_descriptors, scaler = bovw.normalize_descriptors_jointly(all_descriptors)
    
    print("Fitting the codebook")
    kmeans, cluster_centers = bovw._update_fit_codebook(descriptors=all_descriptors)

    print("Computing the bovw histograms")
    bovw_histograms = extract_bovw_histograms(descriptors=all_descriptors, bovw=bovw) 
    
    print("Fitting the classifier")
    classifier = classifier.fit(bovw_histograms, all_labels)

    print("Accuracy on Phase[Train]:", accuracy_score(y_true=all_labels, y_pred=classifier.predict(bovw_histograms)))
    
    return bovw, classifier, scaler


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

    map_classes = {clsi: idx for idx, clsi  in enumerate(os.listdir(ImageFolder))}
    
    dataset :List[Tuple] = []

    for idx, cls_folder in enumerate(os.listdir(ImageFolder)):

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
    bovw = BOVW(codebook_size=2000) # default works the same
    classifier = LogisticRegression(class_weight="balanced")
    # classifier = SVC(kernel='rbf')
    
    bovw, classifier, scaler = train(dataset=data_train, bovw=bovw, classifier=classifier)
    
    test(dataset=data_test, bovw=bovw, classifier=classifier, scaler=scaler)
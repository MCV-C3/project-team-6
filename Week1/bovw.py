import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import os
import glob


from typing import *

from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

class BOVW():
    
    def __init__(
            self,
            *,
            detector_type = "SIFT",
            codebook_size: int = 50,
            descriptor_normalization = None,
            joint_descriptor_normalization = None,
            detector_kwargs: dict = {},
            codebook_kwargs: dict = {},
            dense_kwargs: dict = {},
            dimensionality_reduction = None,
            dimensionality_reduction_kwargs: dict = {}
        ):

        self.dense = False
        if detector_type == 'SIFT':
            self.detector = cv2.SIFT_create(**detector_kwargs)
        elif detector_type == 'AKAZE':
            self.detector = cv2.AKAZE_create(**detector_kwargs)
        elif detector_type == 'ORB':
            self.detector = cv2.ORB_create(**detector_kwargs)
        elif detector_type == 'DSIFT':
            self.dense = True
            self.detector = cv2.SIFT_create(**detector_kwargs)
        else:
            raise ValueError("Detector type must be 'SIFT', 'DSIFT', 'AKAZE', or 'ORB'")
        
        self.codebook_size = codebook_size
        self.codebook_algo = MiniBatchKMeans(n_clusters=self.codebook_size, **codebook_kwargs)
        self.detector_type = detector_type
        self.detector_kwargs = detector_kwargs
        self.dense_kwargs = dense_kwargs

        self.descriptor_normalization = descriptor_normalization
        self.joint_descriptor_normalization = joint_descriptor_normalization
        self.dimensionality_reduction = dimensionality_reduction
        self.dimensionality_reduction_kwargs = dimensionality_reduction_kwargs
        
        self.scaler = None
        
    ## Modify this function in order to be able to create a dense sift
    def _extract_features(self, image: Literal["H", "W", "C"]) -> Tuple:
        if not self.dense:
            return self.detector.detectAndCompute(image, None)
        else:
            return self._extract_dense_features(image)
        
        
    def _extract_dense_features(self, image: Literal["H", "W", "C"]) -> Tuple:
        step = self.dense_kwargs.get("step", 1)
        size = self.dense_kwargs.get("size", 1)
        
        # TODO: Maybe add padding, or use step//2 as padding?
        
        keypoints = [cv2.KeyPoint(x, y, size) 
                for y in range(0, image.shape[0], step)
                for x in range(0, image.shape[1], step)]
        
        keypoints, descriptors = self.detector.compute(image, keypoints)
        
        return keypoints, descriptors
    
    
    def _update_fit_codebook(self, descriptors: Literal["N", "T", "d"])-> Tuple[Type[MiniBatchKMeans],
                                                                               Literal["codebook_size", "d"]]:
        
        all_descriptors = np.vstack(descriptors)

        self.codebook_algo = self.codebook_algo.partial_fit(X=all_descriptors)

        return self.codebook_algo, self.codebook_algo.cluster_centers_
    
    def _compute_codebook_descriptor(self, descriptors: Literal["1 T d"], kmeans: Type[KMeans]) -> np.ndarray:

        visual_words = kmeans.predict(descriptors)
        
        
        # Create a histogram of visual words
        codebook_descriptor = np.zeros(kmeans.n_clusters)
        for label in visual_words:
            codebook_descriptor[label] += 1
        
        # Normalize the histogram (optional)
        codebook_descriptor = codebook_descriptor / np.linalg.norm(codebook_descriptor)
        
        return codebook_descriptor

    def normalize_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        if self.descriptor_normalization is None:
            return descriptors
        
        # cutre
        match self.descriptor_normalization:
            case "L2":
                norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
                norms[norms == 0] = 1
                return descriptors / norms
            case "L1":
                sums = np.sum(descriptors, axis=1, keepdims=True)
                sums[sums == 0] = 1
                return descriptors / sums
            case "Root":
                norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
                norms[norms == 0] = 1
                return np.sqrt(descriptors / norms)
            case _:
                raise ValueError("Invalid normalization.")


    # FIXME: this could be fused with the method above
    def fit_scale_descriptors_jointly(self, all_descriptors: list[np.ndarray]) -> list[np.ndarray]:
        if self.joint_descriptor_normalization is None:
            return all_descriptors

        # cutre
        # parece muy caro normalizar todo esto (?)
        descriptors = np.concat(all_descriptors)
        match self.joint_descriptor_normalization:
            case "MaxAbs":
                self.scaler = MaxAbsScaler()
            case "Standard":
                self.scaler = StandardScaler()
            case "MinMax":
                self.scaler = MinMaxScaler()
            case _:
                raise ValueError("Invalid normalization for all descriptors.")

        self.scaler.fit(descriptors)
        return self.normalize_all_descriptors(all_descriptors)


    def scale_all_descriptors(self, all_descriptors: list[np.ndarray]) -> list[np.ndarray]:
        if self.scaler is None:
            return all_descriptors

        return [self.scaler.transform(descriptors) for descriptors in all_descriptors]


def visualize_bow_histogram(histogram, image_index, output_folder="./test_example.jpg"):
    """
    Visualizes the Bag of Visual Words histogram for a specific image and saves the plot to the output folder.
    
    Args:
        histogram (np.array): BoVW histogram.
        cluster_centers (np.array): Cluster centers (visual words).
        image_index (int): Index of the image for reference.
        output_folder (str): Folder where the plot will be saved.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(histogram)), histogram)
    plt.title(f"BoVW Histogram for Image {image_index}")
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.xticks(range(len(histogram)))
    
    # Save the plot to the output folder
    plot_path = os.path.join(output_folder, f"bovw_histogram_image_{image_index}.png")
    plt.savefig(plot_path)
    
    # Optionally, close the plot to free up memory
    plt.close()

    print(f"Plot saved to: {plot_path}")


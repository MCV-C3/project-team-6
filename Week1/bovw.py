import cv2
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import os
import glob


from typing import *

from sklearn.discriminant_analysis import StandardScaler, LinearDiscriminantAnalysis
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from skimage.feature import fisher_vector, learn_gmm

DetectorType = Literal["SIFT", "AKAZE", "ORB", "DSIFT"]
DescriptorNormalization = Literal["L1", "L2", "Root"]
JointDescriptorNormalization = Literal["MaxAbs", "Standard", "MinMax"]
DimensionalityReduction = Literal["PCA", "SVD", "LDA", "TSNE"]
EncodingMethod = Literal["bovw", "fisher"]

class BOVW():
    """
    Bag of Visual Words (BOVW) implementation for image classification.
    """

    def __init__(
            self,
            *,
            detector_type: DetectorType = "SIFT",
            codebook_size: int = 50,
            descriptor_normalization: Optional[DescriptorNormalization] = None,
            joint_descriptor_normalization: Optional[JointDescriptorNormalization] = None,
            detector_kwargs: dict = {},
            codebook_kwargs: dict = {},
            dense_kwargs: dict = {},
            dimensionality_reduction: Optional[DimensionalityReduction] = None,
            dimensionality_reduction_kwargs: dict = {},
            pyramid_levels: Optional[int] = None,
            encoding_method: EncodingMethod = "bovw",
        ):
        """
        Initialize the BOVW model.

        Args:
            detector_type: Feature detector type. Options: "SIFT", "AKAZE", "ORB", "DSIFT".
            codebook_size: Number of visual words in the codebook (must be >= 2). For Fisher Vectors, this is the number of GMM modes.
            descriptor_normalization: Per-descriptor normalization. Options: "L1", "L2", "Root", or None.
            joint_descriptor_normalization: Joint normalization across all descriptors. Options: "MaxAbs", "Standard", "MinMax", or None.
            detector_kwargs: Additional keyword arguments for the detector.
            codebook_kwargs: Additional keyword arguments for MiniBatchKMeans (BOVW) or GaussianMixture (Fisher).
            dense_kwargs: Dense SIFT parameters (e.g., {"step": 32, "size": 1}).
            dimensionality_reduction: Dimensionality reduction method. Options: "PCA", "SVD", "LDA", "TSNE", or None.
            dimensionality_reduction_kwargs: Additional keyword arguments for the dimensionality reduction method.
            pyramid_levels: Spatial pyramid levels (None for classic encoding, or >= 1 for pyramid).
            encoding_method: Encoding method. Options: "bovw" (Bag of Visual Words), "fisher" (Fisher Vectors).
        """
        if codebook_size < 2:
            raise ValueError("codebook_size must be at least 2")

        if pyramid_levels is not None and pyramid_levels < 1:
            raise ValueError("pyramid_levels must be None or >= 1")

        if detector_type not in get_args(DetectorType):
            raise ValueError(f"detector_type must be one of: {get_args(DetectorType)}. Got: {detector_type}")

        if descriptor_normalization is not None and descriptor_normalization not in get_args(DescriptorNormalization):
            raise ValueError(f"descriptor_normalization must be None or one of: {get_args(DescriptorNormalization)}. Got: {descriptor_normalization}")

        if joint_descriptor_normalization is not None and joint_descriptor_normalization not in get_args(JointDescriptorNormalization):
            raise ValueError(f"joint_descriptor_normalization must be None or one of: {get_args(JointDescriptorNormalization)}. Got: {joint_descriptor_normalization}")

        if dimensionality_reduction is not None and dimensionality_reduction not in get_args(DimensionalityReduction):
            raise ValueError(f"dimensionality_reduction must be None or one of: {get_args(DimensionalityReduction)}. Got: {dimensionality_reduction}")

        if encoding_method not in get_args(EncodingMethod):
            raise ValueError(f"encoding_method must be one of: {get_args(EncodingMethod)}. Got: {encoding_method}")

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
        
        self.codebook_size = codebook_size
        self.encoding_method = encoding_method

        # Initialize encoding algorithm based on method
        if self.encoding_method == "bovw":
            self.codebook_algo = MiniBatchKMeans(n_clusters=self.codebook_size, **codebook_kwargs)
            self.gmm = None
        elif self.encoding_method == "fisher":
            self.codebook_algo = None
            self.gmm = None  # Will be fitted later using learn_gmm
            self.codebook_kwargs = codebook_kwargs

        self.detector_type = detector_type
        self.detector_kwargs = detector_kwargs
        self.dense_kwargs = dense_kwargs

        self.descriptor_normalization = descriptor_normalization
        self.joint_descriptor_normalization = joint_descriptor_normalization
        self.dimensionality_reduction = dimensionality_reduction
        self.dimensionality_reduction_kwargs = dimensionality_reduction_kwargs

        self.scaler = None
        self.dim_reducer = None
        self.pyramid_levels = pyramid_levels
        
    def _extract_features(self, image: Literal["H", "W", "C"]) -> Tuple:
        """
        Extract features from an image using the configured detector.

        Args:
            image: Input image array (H, W, C).

        Returns:
            Tuple of (keypoints, descriptors).
        """
        if not self.dense:
            return self.detector.detectAndCompute(image, None)
        else:
            return self._extract_dense_features(image)
        
        
    def _extract_dense_features(self, image: Literal["H", "W", "C"]) -> Tuple:
        """
        Extract dense SIFT features on a regular grid.

        Args:
            image: Input image array (H, W, C).

        Returns:
            Tuple of (keypoints, descriptors).
        """
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
        """
        Fit the codebook using MiniBatchKMeans on the provided descriptors.

        Args:
            descriptors: List of descriptor arrays (N, T, d).

        Returns:
            Tuple of (fitted kmeans model, cluster centers).
        """
        all_descriptors = np.vstack(descriptors)

        self.codebook_algo = self.codebook_algo.partial_fit(X=all_descriptors)

        return self.codebook_algo, self.codebook_algo.cluster_centers_

    def _update_fit_gmm(self, descriptors: Literal["N", "T", "d"]) -> Type[GaussianMixture]:
        """
        Fit the GMM using learn_gmm on the provided descriptors for Fisher Vectors.

        Args:
            descriptors: List of descriptor arrays (N, T, d).

        Returns:
            Fitted GMM model.
        """
        # learn_gmm expects a list of arrays, NOT a stacked array
        if not isinstance(descriptors, list):
            raise TypeError(f"descriptors must be a list of arrays, got {type(descriptors)}")
        
        all_descriptors = np.vstack(descriptors)
        
        # Convert to float64 for numerical stability
        all_descriptors = all_descriptors.astype(np.float64)
        
        # Adaptive component reduction: if we don't have enough samples per component
        max_components = max(2, all_descriptors.shape[0] // 5)  # At least 5 samples per component
        effective_codebook_size = min(self.codebook_size, max_components)
        
        if effective_codebook_size < self.codebook_size:
            print(f"Warning: Reducing GMM components from {self.codebook_size} to {effective_codebook_size} due to insufficient samples")
        
        if all_descriptors.shape[0] < effective_codebook_size:
            raise ValueError(
                f"Not enough descriptors ({all_descriptors.shape[0]}) "
                f"for {effective_codebook_size} GMM components. Need at least {effective_codebook_size}."
            )
        
        # Verificar que no hay dimensiones con varianza cero
        if np.any(np.std(all_descriptors, axis=0) == 0):
            raise ValueError("Some descriptor dimensions have zero variance. Check normalization.")
        
        # learn_gmm can take a list of arrays directly
        # Ensure covariance_type is 'diag' as required by Fisher vectors
        gm_args = self.codebook_kwargs.copy() if self.codebook_kwargs else {}
        gm_args['covariance_type'] = 'diag'
        gm_args['reg_covar'] = 1e-3
        gm_args['n_init'] = 3

        self.gmm = learn_gmm(descriptors, n_modes=effective_codebook_size, gm_args=gm_args)

        return self.gmm
    
    def _compute_codebook_descriptor(self, descriptors: Literal["1 T d"], keypoints: list[cv2.KeyPoint], kmeans: Type[KMeans], image_size: Tuple[int, int]) -> np.ndarray:
        """
        Compute the encoded descriptor for an image (BOVW or Fisher Vector).

        Args:
            descriptors: Feature descriptors (T, d).
            keypoints: Keypoints corresponding to the descriptors.
            kmeans: Fitted KMeans model (for BOVW) or GMM (for Fisher).
            image_size: Image dimensions (height, width).

        Returns:
            Encoded descriptor (BOVW histogram or Fisher vector, with or without spatial pyramid).
        """
        if self.encoding_method == "bovw":
            if self.pyramid_levels is None:
                return self._compute_codebook_descriptor_classic(descriptors=descriptors, kmeans=kmeans)
            else:
                return self._compute_codebook_descriptor_pyramid(descriptors=descriptors, keypoints=keypoints, kmeans=kmeans, image_size=image_size)
        elif self.encoding_method == "fisher":
            if self.pyramid_levels is None:
                return self._compute_fisher_vector_classic(descriptors=descriptors, gmm=kmeans)  # kmeans parameter holds GMM for fisher
            else:
                return self._compute_fisher_vector_pyramid(descriptors=descriptors, keypoints=keypoints, gmm=kmeans, image_size=image_size)
    
    def _compute_codebook_descriptor_classic(self, descriptors: Literal["1 T d"], kmeans: Type[KMeans]) -> np.ndarray:
        """
        Compute classic BOVW histogram (no spatial pyramid).

        Args:
            descriptors: Feature descriptors (T, d).
            kmeans: Fitted KMeans model.

        Returns:
            Normalized histogram of visual words.
        """
        visual_words = kmeans.predict(descriptors)

        codebook_descriptor = np.zeros(kmeans.n_clusters)
        for label in visual_words:
            codebook_descriptor[label] += 1

        codebook_descriptor = codebook_descriptor / np.linalg.norm(codebook_descriptor)

        return codebook_descriptor

    def _compute_codebook_descriptor_pyramid(self, descriptors: Literal["1 T d"], keypoints: list[cv2.KeyPoint], kmeans: Type[KMeans], image_size: Tuple[int, int]) -> np.ndarray:
        """
        Compute spatial pyramid BOVW descriptor.

        Args:
            descriptors: Feature descriptors (T, d).
            keypoints: Keypoints corresponding to the descriptors.
            kmeans: Fitted KMeans model.
            image_size: Image dimensions (height, width).

        Returns:
            Concatenated normalized histograms for all pyramid levels and regions.
        """
        height, width = image_size

        visual_words = kmeans.predict(descriptors)

        kp_coords = np.array([kp.pt for kp in keypoints])

        all_histograms = []

        for level in range(1, self.pyramid_levels + 1):
            grid_size = level
            cell_height = height / grid_size
            cell_width = width / grid_size

            for row in range(grid_size):
                for col in range(grid_size):
                    y_min = row * cell_height
                    y_max = (row + 1) * cell_height
                    x_min = col * cell_width
                    x_max = (col + 1) * cell_width

                    mask = (
                        (kp_coords[:, 0] >= x_min) & (kp_coords[:, 0] < x_max) &
                        (kp_coords[:, 1] >= y_min) & (kp_coords[:, 1] < y_max)
                    )

                    region_words = visual_words[mask]

                    histogram = np.zeros(kmeans.n_clusters)
                    for word in region_words:
                        histogram[word] += 1

                    norm = np.linalg.norm(histogram)
                    if norm > 0:
                        histogram = histogram / norm

                    all_histograms.append(histogram)

        return np.concatenate(all_histograms)

    def _compute_fisher_vector_classic(self, descriptors: Literal["1 T d"], gmm: Type[GaussianMixture]) -> np.ndarray:
        """
        Compute classic Fisher Vector (no spatial pyramid).

        Args:
            descriptors: Feature descriptors (T, d).
            gmm: Fitted GMM model.

        Returns:
            Fisher vector encoding.
        """
        fv = fisher_vector(descriptors, gmm, improved=True)
        return fv

    def _compute_fisher_vector_pyramid(self, descriptors: Literal["1 T d"], keypoints: list[cv2.KeyPoint], gmm: Type[GaussianMixture], image_size: Tuple[int, int]) -> np.ndarray:
        """
        Compute spatial pyramid Fisher Vector descriptor.

        Args:
            descriptors: Feature descriptors (T, d).
            keypoints: Keypoints corresponding to the descriptors.
            gmm: Fitted GMM model.
            image_size: Image dimensions (height, width).

        Returns:
            Concatenated Fisher vectors for all pyramid levels and regions.
        """
        height, width = image_size
        kp_coords = np.array([kp.pt for kp in keypoints])

        all_fisher_vectors = []

        for level in range(1, self.pyramid_levels + 1):
            grid_size = level
            cell_height = height / grid_size
            cell_width = width / grid_size

            for row in range(grid_size):
                for col in range(grid_size):
                    y_min = row * cell_height
                    y_max = (row + 1) * cell_height
                    x_min = col * cell_width
                    x_max = (col + 1) * cell_width

                    mask = (
                        (kp_coords[:, 0] >= x_min) & (kp_coords[:, 0] < x_max) &
                        (kp_coords[:, 1] >= y_min) & (kp_coords[:, 1] < y_max)
                    )

                    region_descriptors = descriptors[mask]

                    # If no descriptors in this region, use zeros
                    if len(region_descriptors) == 0:
                        # Fisher vector dimension: 2*K*D + K (gradients w.r.t. weights, means, covariances)
                        fv_dim = 2 * gmm.n_components * descriptors.shape[1] + gmm.n_components
                        fv = np.zeros(fv_dim)
                    else:
                        fv = fisher_vector(region_descriptors, gmm, improved=True)

                    all_fisher_vectors.append(fv)

        return np.concatenate(all_fisher_vectors)

    def normalize_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Normalize individual descriptors using the configured normalization method.

        Args:
            descriptors: Input descriptors array.

        Returns:
            Normalized descriptors.
        """
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
                norms = np.sum(descriptors, axis=1, keepdims=True)
                norms[norms == 0] = 1
                return np.sqrt(descriptors / norms)
            case _:
                raise ValueError("Invalid normalization.")


    # FIXME: this could be fused with the method above
    def fit_scale_descriptors_jointly(self, all_descriptors: list[np.ndarray]) -> list[np.ndarray]:
        """
        Fit and apply joint normalization across all descriptors.

        Args:
            all_descriptors: List of descriptor arrays.

        Returns:
            Normalized descriptor arrays.
        """
        if self.joint_descriptor_normalization is None:
            return all_descriptors

        # cutre
        # parece muy caro normalizar todo esto (?)
        descriptors = np.concatenate(all_descriptors, axis=0)
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
        return self.scale_all_descriptors(all_descriptors)


    def scale_all_descriptors(self, all_descriptors: list[np.ndarray]) -> list[np.ndarray]:
        """
        Apply the fitted scaler to normalize descriptors.

        Args:
            all_descriptors: List of descriptor arrays.

        Returns:
            Scaled descriptor arrays.
        """
        if self.scaler is None:
            return all_descriptors

        return [self.scaler.transform(descriptors) for descriptors in all_descriptors]


    def prepare_labels_for_lda(self, all_descriptors: list[np.ndarray], image_labels: list[int]) -> np.ndarray:
        """
        Prepare labels for LDA by repeating each image label for all its descriptors.

        Args:
            all_descriptors: List of descriptor arrays (N images, each with T_i descriptors of dimension d).
            image_labels: List of image labels (N labels, one per image).

        Returns:
            Array of labels where each descriptor gets the label of its parent image.
            Shape: (total_descriptors,)
        """
        if len(all_descriptors) != len(image_labels):
            raise ValueError(f"Number of descriptor arrays ({len(all_descriptors)}) must match number of labels ({len(image_labels)})")

        descriptor_labels = []
        for descriptors, label in zip(all_descriptors, image_labels):
            num_descriptors = descriptors.shape[0]
            descriptor_labels.extend([label] * num_descriptors)

        return np.array(descriptor_labels)


    def fit_reduce_dimensionality(self, all_descriptors: list[np.ndarray], labels: Optional[list[int]] = None) -> list[np.ndarray]:
        """
        Fit and apply dimensionality reduction to descriptors.

        Args:
            all_descriptors: List of descriptor arrays.
            labels: Optional labels for supervised methods (e.g., LDA).

        Returns:
            Dimensionality-reduced descriptor arrays.
        """
        if self.dimensionality_reduction is None:
            return all_descriptors

        descriptors = np.concatenate(all_descriptors, axis=0)
        match self.dimensionality_reduction:
            case "PCA":
                self.dim_reducer = PCA(**self.dimensionality_reduction_kwargs)
            case "SVD":
                self.dim_reducer = TruncatedSVD(**self.dimensionality_reduction_kwargs)
            case "LDA":
                # LDA because it appears on the slides
                if labels is None:
                    raise ValueError("LDA requires labels.")
                self.dim_reducer = LinearDiscriminantAnalysis(**self.dimensionality_reduction_kwargs)
            case "TSNE":
                self.dim_reducer = TSNE(**self.dimensionality_reduction_kwargs)
            case _:
                raise ValueError("Invalid dimensionality reduction method. Choose from: PCA, SVD, LDA, TSNE")

        if self.dimensionality_reduction == "LDA":
            # LDA needs labels during fit!!!
            # Prepare labels for each descriptor (not just each image)
            descriptor_labels = self.prepare_labels_for_lda(all_descriptors, labels)
            self.dim_reducer.fit(descriptors, descriptor_labels)
        else:
            self.dim_reducer.fit(descriptors)

        return self.reduce_dimensionality(all_descriptors)


    def reduce_dimensionality(self, all_descriptors: list[np.ndarray]) -> list[np.ndarray]:
        """
        Apply the fitted dimensionality reducer to descriptors.

        Args:
            all_descriptors: List of descriptor arrays.

        Returns:
            Dimensionality-reduced descriptor arrays.
        """
        if self.dim_reducer is None:
            return all_descriptors

        return [self.dim_reducer.transform(descriptors) for descriptors in all_descriptors]


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


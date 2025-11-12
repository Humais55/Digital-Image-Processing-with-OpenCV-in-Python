import numpy as np
import cv2
import rasterio
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial import distance

# Load and normalize 2014 & 2025 bands (B2-B7)
def load_bands(path):
    bands = []
    meta = None
    for i in range(2, 8):
        band_path = f"{path}B{i}.tif"
        with rasterio.open(band_path) as src:
            band = src.read(1).astype('float32')
            band = np.nan_to_num(band, nan=0.0)
            if meta is None:
                meta = src.meta.copy()
            band_norm = cv2.normalize(band, None, 0, 1, cv2.NORM_MINMAX)
            bands.append(band_norm)
    bands = np.stack(bands, axis=-1)
    return bands, meta

path_2014 = "D:/Courses/Digital-Image-Processing-with-OpenCV-in-Python/img/UC_Mian_Landsat_CloudFree_B1_B7_2014/"
path_2025 = "D:/Courses/Digital-Image-Processing-with-OpenCV-in-Python/img/UC_Mian_Landsat_CloudFree_B1_B7_2025/"

landsat_2014, meta_2014 = load_bands(path_2014)
landsat_2025, meta_2025 = load_bands(path_2025)

# Flatten for K-Means
X_2014 = landsat_2014.reshape((-1,6)).astype(np.float32)
X_2025 = landsat_2025.reshape((-1,6)).astype(np.float32)

# Run K-Means on 2014
K = 6
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1)
ret_2014, labels_2014, centers_2014 = cv2.kmeans(X_2014, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
labels_img_2014 = labels_2014.reshape((landsat_2014.shape[0], landsat_2014.shape[1]))

# Assign 2025 pixels to nearest 2014 cluster center
distances = distance.cdist(X_2025, centers_2014, 'euclidean')  # shape: (num_pixels, K)
labels_2025 = np.argmin(distances, axis=1)
labels_img_2025 = labels_2025.reshape((landsat_2025.shape[0], landsat_2025.shape[1]))

# Define colors
colors = [
    [0.8, 0.1, 0.1],   # red
    [0.1, 0.8, 0.1],   # green
    [0.1, 0.1, 0.8],   # blue
    [0.8, 0.8, 0.1],   # yellow
    [0.8, 0.1, 0.8],   # magenta
    [0.1, 0.8, 0.8]    # cyan
]
cmap = ListedColormap(colors[:K])

# Visualization
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.imshow(labels_img_2014, cmap=cmap)
plt.title("Landsat 2014 Unsupervised Classification (K-Means)")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(labels_img_2025, cmap=cmap)
plt.title("Landsat 2025 Unsupervised Classification (K-Means)")
plt.axis('off')
plt.show()
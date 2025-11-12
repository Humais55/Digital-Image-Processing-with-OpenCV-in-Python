import numpy as np
import cv2
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Function to display using OpenCV
def wind(image, title='Landsat 8'):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Input Folder
path_2014 = "D:/Courses/Digital-Image-Processing-with-OpenCV-in-Python/img/UC_Mian_Landsat_CloudFree_B1_B7_2014/"
path_2025 = "D:/Courses/Digital-Image-Processing-with-OpenCV-in-Python/img/UC_Mian_Landsat_CloudFree_B1_B7_2025/"

landsat_2014 = []
meta_2014 = None

landsat_2025 = []
meta_2025 = None

# Select meaningful bands (B2–B7)
for i in range(2, 8):
    band_path = f"{path_2014}B{i}.tif"
    with rasterio.open(band_path) as src:
        band = src.read(1).astype('float32')
        band = np.nan_to_num(band, nan=0.0)

        # Save metadata from first band for GeoTIFF output
        if meta_2014 is None:
            meta_2014 = src.meta.copy()

        # Normalize band between 0–1
        band_norm = cv2.normalize(band, None, 0, 1, cv2.NORM_MINMAX)
        landsat_2014.append(band_norm)

for i in range(2, 8):
    band_path = f"{path_2025}B{i}.tif"
    with rasterio.open(band_path) as src:
        band = src.read(1).astype('float32')
        band = np.nan_to_num(band, nan=0.0)

        # Save metadata from first band for GeoTIFF
        if meta_2025 is None:
            meta_2025 = src.meta.copy()

        # Normalize band between 0–1
        band_norm = cv2.normalize(band, None, 0, 1, cv2.NORM_MINMAX)
        landsat_2025.append(band_norm)

# Stack selected bands
landsat_2014 = np.stack(landsat_2014, axis=-1)
print("Bands stacked (2014):", landsat_2014.shape)  # (rows, cols, 6)

landsat_2025 = np.stack(landsat_2025, axis=-1)
print("Bands stacked (2025):", landsat_2025.shape)  # (rows, cols, 6)

# Flatten for K-Means
X_2014 = landsat_2014.reshape((-1, 6))
X_2014 = np.float32(X_2014)

X_2025 = landsat_2025.reshape((-1, 6))
X_2025 = np.float32(X_2025)

# Define K-Means criteria
K = 6
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1)

# Run K-Means clustering
print("Running K-Means for 2014 image...")
ret, labels_2014, centers = cv2.kmeans(X_2014, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

print("Running K-Means for 2025 image...")
ret, labels_2025, centers = cv2.kmeans(X_2025, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Reshape classification result
labels_img_2014 = labels_2014.reshape((landsat_2014.shape[0], landsat_2014.shape[1]))
labels_img_2025 = labels_2025.reshape((landsat_2025.shape[0], landsat_2025.shape[1]))

# Visualization using matplotlib
colors = [
    [0.8, 0.1, 0.1],   # red
    [0.1, 0.8, 0.1],   # green
    [0.1, 0.1, 0.8],   # blue
    [0.8, 0.8, 0.1],   # yellow
    [0.8, 0.1, 0.8],   # magenta
    [0.1, 0.8, 0.8]    # cyan
]
cmap = ListedColormap(colors[:K])

"""
plt.figure(figsize=(8, 6))
plt.imshow(labels_img_2014, cmap=cmap)
plt.title("Landsat 2014 Unsupervised Classification (K-Means)")
plt.axis('off')
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(labels_img_2025, cmap=cmap)
plt.title("Landsat 2025 Unsupervised Classification (K-Means)")
plt.axis('off')
plt.show()
"""

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

"""
# === Convert to uint8 for saving ===
classified_uint8_2014 = labels_img_2014.astype(np.uint8)

# === Save classification as GeoTIFF ===
meta_2014.update({
    "count": 1,
    "dtype": "uint8"
})

output_path_2014 = path_2014 + "Landsat_KMeans_Classification.tif"

with rasterio.open(output_path_2014, 'w', **meta_2014) as dst:
    dst.write(classified_uint8_2014, 1)

print(f"✅ 2014 Classification saved as GeoTIFF:\n{output_path}")

classified_uint8_2025 = labels_img_2025.astype(np.uint8)

meta_2025.update({
    "count": 1,
    "dtype": "uint8"
})

output_path_2025 = path_2025 + "Landsat_KMeans_Classification_2025.tif"

with rasterio.open(output_path_2025, 'w', **meta_2025) as dst:
    dst.write(classified_uint8_2025, 1)

print(f"✅ 2025 Classification saved as GeoTIFF:\n{output_path_2025}")
"""
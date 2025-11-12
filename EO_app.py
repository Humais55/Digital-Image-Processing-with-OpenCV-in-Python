# Optimal Importing of Multiple Bands

import numpy as np
import cv2 
import rasterio
from matplotlib import pyplot as plt

def wind(image):
    cv2.namedWindow('Landsat_8', cv2.WINDOW_NORMAL)
    cv2.imshow('Landsat 8', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

path = "D:/Courses/Digital-Image-Processing-with-OpenCV-in-Python/img/UC_Mian_Landsat_CloudFree_B1_B7_2018/"

landsat = []

for i in range(1, 8):
    band_path = f"{path}B{i}.tif"
    with rasterio.open(band_path) as src:
        band = src.read(1).astype('float32')

        # Replace NaNs with 0
        band = np.nan_to_num(band, nan=0.0)

        # Normalize to 0–255 and convert to uint8
        band_8bit = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX)
        band_8bit = np.clip(band_8bit, 0, 255).astype(np.uint8)

        landsat.append(band_8bit)

#wind(landsat[4])
rgb = cv2.merge((landsat[0], landsat[1], landsat[2], landsat[3], landsat[4], landsat[5], landsat[6]))  # B4, B3, B2

# CLassification and extraction of final image
rgba = rgb.reshape((-1, 7))
crit = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,1)
ret, label, center = cv2.kmeans(rgba, 6, None, crit, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((rgb.shape))

res3 = res2[:][:,:, :3]
wind(res3)


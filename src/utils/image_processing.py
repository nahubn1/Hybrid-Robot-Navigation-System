import numpy as np
from scipy import ndimage


def distance_transform(grid: np.ndarray) -> np.ndarray:
    obstacles = grid != 0
    return ndimage.distance_transform_edt(~obstacles)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    structure = ndimage.generate_binary_structure(2, 1)
    return ndimage.binary_dilation(mask, structure=structure, iterations=radius)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(img.astype(float), sigma=sigma)

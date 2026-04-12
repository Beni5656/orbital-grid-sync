import json
from pathlib import Path

import albumentations as A
import numpy as np
import rasterio
import rasterio.features
import rasterio.windows
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
import torch
from torch.utils.data import Dataset

def reproject_to_crs(src_path: str, dst_path: str, target_crs: str = "EPSG:4326"):
    """
    Reproject a GeoTIFF image to a common crs for accurate comparisons
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({"crs": target_crs, "transform": transform, "width": width, "height": height})
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source = rasterio.band(src, band_idx),
                    destination = rasterio.band(dst, band_idx),
                    src_transform = src.transform,
                    src_crs = src.crs,
                    dst_transform = transform,
                    dst_crs = target_crs,
                    resampling = Resampling.bilinear,
                )
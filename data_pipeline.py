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

def tile_geotiff(src_path: str, output_dir: str, tile_size: int=512, overlap: int=64, min_valid_ratio: float=0.8) -> list:
    """
    Extract tiles from a large GeoTIFF 
    """
    output_dir = Path(output_dir);
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    stride = tile_size - overlap

    with rasterio.open(src_path) as src:
        H, W = src.height, src.width
        nodata = src.nodata

        for row_start in range(0, H - tile_size + 1, stride):
            for col_start in range(0, W - tile_size + 1, stride):
                window = rasterio.windows.Window(col_start, row_start, tile_size, tile_size)
                tile = src.read(window=window)

                if nodata is not None:
                    valid_ratio = (tile != nodata).mean()
                    if valid_ratio < min_valid_ratio:
                        continue
                
                bounds = rasterio.windows.bounds(window, src.transform)
                tile_id = f"r{row_start:05d}_c{col_start:05d}"
                np.save(output_dir / f"{tile_id}.npy", tile.astype(np.float32))

                manifest.append({
                    "tile_id": tile_id,
                    "row": row_start,
                    "col": col_start,
                    "bounds": list(bounds),   
                    "crs": str(src.crs),
                })
        
        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Tiled {len(manifest)} valid patches from {Path(src_path).name}")
        return manifest
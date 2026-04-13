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

class PercentileNormalizer:
    """
    Used to normalize images, pixel values less than 2nd percentile and 98th percentile and treated as the min and max
    """
    def init(self, p_low: float=2, p_high: float=98):
        self.p_low = p_low
        self.p_high = p_high
        self.lo = None
        self.hi = None
    
    def fit(self, tiles: np.ndarray) -> "PercentileNormalizer":
        flat = tiles.reshape(tiles.shape[1], -1)
        self.lo = np.percentile(flat, self.p_low, axis=1)
        self.hi = np.percentile(flat, self.p_high, axis=1)
        return self
    
    def save(self, path: str):
        np.savez(path, lo=self.lo, hi=self.hi)

    def load(self, path: str) -> "PercentileNormalizer":
        data = np.load(path)
        self.lo, self.hi = data['lo'], data['hi']
        return self
    
    def __call__(self, tile: np.ndarray) -> np.ndarray:
        """tile: (C, H, W) → [0, 1] float32"""
        lo = self.lo[:, None, None]
        hi = self.hi[:, None, None]
        return np.clip((tile - lo) / (hi - lo + 1e-8), 0.0, 1.0)    
    
def get_train_transforms(tile_size: int = 512) -> A.Compose:
    """
    Applies Randomness to Each Image
    """
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(tile_size, tile_size),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=1.0),
            A.RandomBrightnessContrast(p=1.0),
        ], p=0.5),
    ], additional_targets={"image2": "image"})


def get_val_transforms(tile_size: int = 512) -> A.Compose:
    """
    Always crops in the center spot 
    """
    return A.Compose([
        A.CenterCrop(tile_size, tile_size),
    ], additional_targets={"image2": "image"})

class ChangeDetectionDataset(Dataset):
    """
    Pytorch Dataset for multi-temporal change detection
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        tile_size: int = 512,
        normalizer: PercentileNormalizer = None,
    ):
        self.root = Path(root)
        self.tile_size = tile_size
        self.normalizer = normalizer

        self.t1_files   = sorted((self.root / "t1").glob("*.npy"))
        self.t2_files   = sorted((self.root / "t2").glob("*.npy"))
        self.mask_files = sorted((self.root / "masks").glob("*.npy"))

        assert len(self.t1_files) == len(self.t2_files) == len(self.mask_files), \
            "Mismatch between T1, T2, and mask file counts."

        self.transforms = (
            get_train_transforms(tile_size) if split == "train"
            else get_val_transforms(tile_size)
        )

    def __len__(self):
        return len(self.t1_files)

    def __getitem__(self, idx):
        t1   = np.load(self.t1_files[idx])    
        t2   = np.load(self.t2_files[idx])
        mask = np.load(self.mask_files[idx])   

        if self.normalizer:
            t1 = self.normalizer(t1)
            t2 = self.normalizer(t2)

        result = self.transforms(
            image=t1.transpose(1, 2, 0),
            image2=t2.transpose(1, 2, 0),
            mask=mask,
        )

        return {
            "t1":     torch.from_numpy(result["image"].transpose(2, 0, 1)),
            "t2":     torch.from_numpy(result["image2"].transpose(2, 0, 1)),
            "mask":   torch.from_numpy(result["mask"]).long(),
            "t1_path": str(self.t1_files[idx]),
        }

def mask_to_geojson(mask: np.ndarray, src_path: str, output_path: str, threshold: float = 0.5, min_area_m2: float = 100.0):
    """
    Converts a predicted probability mask into a georeferenced GeoJSON file
    """
    try:
        import geopandas as gpd
        from shapely.geometry import shape
    except ImportError:
        raise ImportError("pip install geopandas shapely")

    binary = (mask > threshold).astype(np.uint8)

    with rasterio.open(src_path) as src:
        shapes_gen = rasterio.features.shapes(binary, transform=src.transform)
        geometries = [
            {"geometry": shape(geom), "probability": float(val)}
            for geom, val in shapes_gen if val == 1
        ]
        crs = src.crs

    gdf = gpd.GeoDataFrame(geometries, crs=crs)
    gdf = gdf[gdf.geometry.area > min_area_m2 * 1e-10]  
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"Saved {len(gdf)} change polygons → {output_path}")

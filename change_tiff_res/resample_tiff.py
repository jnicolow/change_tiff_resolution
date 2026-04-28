from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling

InterpolationMethod = Literal["bilinear", "bicubic"]

_RASTERIO_INTERPOLATION = {
    "bilinear": Resampling.bilinear,
    "bicubic": Resampling.cubic,
}


def _validate_resolution(desired_resolution: tuple[int, int]) -> tuple[int, int]:
    if len(desired_resolution) != 2:
        raise ValueError("desired_resolution must be a tuple of (height, width).")

    height, width = desired_resolution
    if height <= 0 or width <= 0:
        raise ValueError("desired_resolution values must be positive integers.")

    return int(height), int(width)


def resample_tiff(
    tiff_path: str | Path,
    desired_resolution: tuple[int, int],
    technique: InterpolationMethod = "bilinear",
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Load a TIFF, upsample/downsample to desired_resolution, and optionally save.

    Args:
        tiff_path: Input TIFF path.
        desired_resolution: Output size as (height, width).
        technique: Interpolation method ("bilinear" or "bicubic").
        save_path: Optional output path. If provided, writes the resampled TIFF.

    Returns:
        Resampled image as a NumPy array.
    """
    if technique not in _RASTERIO_INTERPOLATION:
        raise ValueError("technique must be one of: 'bilinear', 'bicubic'.")

    target_h, target_w = _validate_resolution(desired_resolution)
    tiff_path = Path(tiff_path)

    with rasterio.open(tiff_path) as src:
        resampled = src.read(
            out_shape=(src.count, target_h, target_w),
            resampling=_RASTERIO_INTERPOLATION[technique],
        )

        # Convert rasterio's (bands, height, width) into plot-friendly output.
        if src.count == 1:
            result = resampled[0]
        else:
            result = np.transpose(resampled, (1, 2, 0))

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            scale_x = src.width / target_w
            scale_y = src.height / target_h
            transform = src.transform * Affine.scale(scale_x, scale_y)
            profile = src.profile.copy()
            profile.update(
                height=target_h,
                width=target_w,
                transform=transform,
            )

            with rasterio.open(save_path, "w", **profile) as dst:
                dst.write(resampled)

    return result


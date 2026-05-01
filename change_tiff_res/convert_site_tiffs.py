"""Batch upsample GeoTIFFs per site (bilinear and bicubic), mirroring change_tiff_res_test."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Literal

import rasterio

from change_tiff_res import resample_tiff

InterpolationMethod = Literal["bilinear", "bicubic"]

_DEFAULT_TECHNIQUES: tuple[InterpolationMethod, ...] = ("bilinear", "bicubic")


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _collect_tiffs(site_root: Path) -> list[Path]:
    found: set[Path] = set()
    for pattern in ("**/*.tif", "**/*.tiff"):
        found.update(site_root.glob(pattern))
    return sorted(found)


def convert_sites_to_higher_resolution(
    source_dir: str | Path,
    site_names: Iterable[str],
    *,
    output_root: str | Path | None = None,
    upsample_scale: int = 2,
    techniques: tuple[InterpolationMethod, ...] = _DEFAULT_TECHNIQUES,
) -> list[Path]:
    """For each site, load every TIFF under ``source_dir/<site>/`` and save upsampled copies.

    Output layout::

        <output_root>/<site_name>/<technique>/<relative_path_under_site>

    ``relative_path_under_site`` preserves subfolders (e.g. ``L5/scene.tif``).

    Args:
        source_dir: Directory that contains one folder per site name.
        site_names: Site folder names to process (must exist under ``source_dir``).
        output_root: Root for outputs. Defaults to ``<this repo>/data/sat_images``.
        upsample_scale: Integer scale factor applied to height and width (same as test notebook).
        techniques: Which interpolations to run (default: bilinear and bicubic).

    Returns:
        List of paths written.
    """
    if upsample_scale < 1:
        raise ValueError("upsample_scale must be >= 1.")

    source_dir = Path(source_dir)
    if output_root is None:
        output_root = _project_root() / "data" / "sat_images"
    output_root = Path(output_root)

    written: list[Path] = []

    for site in site_names:
        site = site.strip()
        print(f'processing: {site}')
        if not site:
            continue

        site_src = source_dir / site
        if not site_src.is_dir():
            warnings.warn(f"Skipping missing site directory: {site_src}", stacklevel=2)
            continue

        tiffs = _collect_tiffs(site_src)
        if not tiffs:
            warnings.warn(f"No TIFF files under {site_src}", stacklevel=2)
            continue

        for tiff_path in tiffs:
            rel = tiff_path.relative_to(site_src)
            with rasterio.open(tiff_path) as src:
                h, w = src.height, src.width
            desired = (h * upsample_scale, w * upsample_scale)

            for technique in techniques:
                if technique not in ("bilinear", "bicubic"):
                    raise ValueError("technique must be 'bilinear' or 'bicubic'.")
                out_path = output_root / site / technique / rel
                resample_tiff(
                    tiff_path=tiff_path,
                    desired_resolution=desired,
                    technique=technique,
                    save_path=out_path,
                )
                written.append(out_path)

    return written


__all__ = ["convert_sites_to_higher_resolution"]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Upsample all TIFFs per site with bilinear and bicubic."
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Parent directory containing site subfolders (e.g. .../sat_images).",
    )
    parser.add_argument(
        "sites",
        help="Comma-separated site folder names.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root (default: <repo>/data/sat_images).",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Integer upsample factor for height and width (default: 2).",
    )
    args = parser.parse_args()
    site_names = [s.strip() for s in args.sites.split(",") if s.strip()]
    paths = convert_sites_to_higher_resolution(
        args.source_dir,
        site_names,
        output_root=args.output_root,
        upsample_scale=args.scale,
    )
    print(f"Wrote {len(paths)} file(s).")


if __name__ == "__main__":
    main()

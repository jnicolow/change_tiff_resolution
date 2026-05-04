"""Two-row figure: (1) native-resolution crops; (2) same scene on the SEN2SR ~2.5 m pixel grid + center zoom.

Row 1: original 10 m crop, bilinear/bicubic at ~5 m (×2 pipeline), SEN2SR at ~2.5 m (×4 model).

Row 2: same grid alignment; crop center follows NDWI/Otsu **shoreline nearest the patch center** (not
the ocean-wide boundary centroid). When ``zoom2_focus='shoreline'``, the **LR window** is also chosen
from a coarse full-scene NDWI/Otsu map so row 1 usually includes coast, not open ocean.

Uses the same triplet discovery as ``water_index_hr_comparison.ipynb``. SEN2SR uses one ``model(batch)``
call on a 128×128 LR crop (no ``predict_large`` tiling).
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlstac
import numpy as np
import rasterio
from rasterio.enums import Resampling
import torch
import torch.nn.functional as F
from rasterio.windows import Window


@dataclass
class ScenePaths:
    site: str
    rel: Path
    original: Path
    bilinear: Path
    bicubic: Path


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _band_aliases() -> Dict[str, Tuple[str, ...]]:
    return {
        "blue": ("blue", "b", "band2", "b2"),
        "green": ("green", "g", "band3", "b3"),
        "red": ("red", "r", "band4", "b4"),
        "nir": ("nir", "nearinfrared", "nearir", "band5", "b5", "nir08", "nir1"),
    }


def extract_band_names(src: rasterio.io.DatasetReader) -> List[str]:
    names: List[str] = []
    for i in range(1, src.count + 1):
        desc = src.descriptions[i - 1] if src.descriptions else None
        tags = src.tags(i)
        tag_name = tags.get("name") or tags.get("band_name") or tags.get("long_name")
        chosen = (desc or tag_name or f"band{i}").strip()
        names.append(chosen)
    return names


def find_band_index(band_names: List[str], key: str) -> int:
    aliases = tuple(_normalize_name(a) for a in _band_aliases()[key])
    normalized = [_normalize_name(n) for n in band_names]
    for idx, n in enumerate(normalized):
        if n in aliases:
            return idx
    for idx, n in enumerate(normalized):
        if any(a in n for a in aliases):
            return idx
    raise ValueError(f"Could not find band '{key}' in names: {band_names}")


def to_reflectance_physical(data: np.ndarray) -> np.ndarray:
    mx = float(np.nanmax(data))
    if mx > 10:
        return (data / 10000.0).astype(np.float32)
    return np.clip(data, 0.0, 1.0).astype(np.float32)


def read_rgbn_window(path: Path, window: Window) -> np.ndarray:
    """Read (4, H, W) RGB+NIR reflectance from a pixel window."""
    with rasterio.open(path) as src:
        if src.count < 4:
            raise ValueError(f"Need ≥4 bands, got {src.count} for {path}")
        data = src.read(window=window).astype(np.float32)
        names = extract_band_names(src)

    ri = find_band_index(names, "red")
    gi = find_band_index(names, "green")
    bi = find_band_index(names, "blue")
    ni = find_band_index(names, "nir")
    stacked = np.stack([data[ri], data[gi], data[bi], data[ni]], axis=0)
    return to_reflectance_physical(stacked)


def center_lr_window(height: int, width: int, side: int) -> Window:
    row_off = max(0, (height - side) // 2)
    col_off = max(0, (width - side) // 2)
    h = min(side, height - row_off)
    w = min(side, width - col_off)
    return Window(col_off, row_off, w, h)


def pick_lr_window_toward_shoreline(
    path: Path,
    lr_patch: int,
    *,
    coarse_factor: int = 24,
) -> Window:
    """Place an ``lr_patch`` window so it covers NDWI/Otsu shoreline near the scene center.

    Uses one or more coarse full-scene reads (cheap). If the first grid misses land/water mix
    (common with very coarse downsampling on narrow coasts), finer grids are tried before falling
    back to geometric center.
    """
    raw_factors = [
        coarse_factor,
        max(8, (coarse_factor * 2 + 1) // 3),
        max(8, coarse_factor // 2),
        max(6, coarse_factor // 3),
        8,
        6,
    ]
    factors: list[int] = []
    for f in raw_factors:
        f = int(max(6, f))
        if f not in factors:
            factors.append(f)

    with rasterio.open(path) as src:
        h, w = src.height, src.width
        names = extract_band_names(src)
        ri = find_band_index(names, "red")
        gi = find_band_index(names, "green")
        bi = find_band_index(names, "blue")
        ni = find_band_index(names, "nir")

        for cf in factors:
            ch = max(48, h // cf)
            cw = max(48, w // cf)
            ch = min(ch, h)
            cw = min(cw, w)
            planes = [
                src.read(b, out_shape=(ch, cw), resampling=Resampling.average)
                for b in (ri + 1, gi + 1, bi + 1, ni + 1)
            ]
            data = np.stack(planes, axis=0).astype(np.float32)

            stack = to_reflectance_physical(data)
            ndwi = ndwi_mcfeeters(stack)
            if not np.any(np.isfinite(ndwi)):
                continue

            t = otsu_threshold(ndwi)
            water = (ndwi > t) & np.isfinite(ndwi)
            frac = float(np.mean(water))
            if frac < 0.02 or frac > 0.98:
                continue

            ero = _binary_erode_3x3(water)
            shore = (water & ~ero) | (
                (~water) & np.isfinite(ndwi) & ~_binary_erode_3x3((~water) & np.isfinite(ndwi))
            )
            if not np.any(shore):
                continue

            ys, xs = np.nonzero(shore)
            pr, pc = (ch - 1) / 2.0, (cw - 1) / 2.0
            d = (ys.astype(np.float64) - pr) ** 2 + (xs.astype(np.float64) - pc) ** 2
            j = int(np.argmin(d))
            cy_c, cx_c = int(ys[j]), int(xs[j])

            cr_full = (cy_c + 0.5) * h / ch - 0.5
            cc_full = (cx_c + 0.5) * w / cw - 0.5

            col_off = int(np.clip(round(cc_full - lr_patch / 2), 0, max(0, w - lr_patch)))
            row_off = int(np.clip(round(cr_full - lr_patch / 2), 0, max(0, h - lr_patch)))
            return Window(col_off, row_off, lr_patch, lr_patch)

    return center_lr_window(h, w, lr_patch)


def scale_window(win: Window, scale: int) -> Window:
    return Window(
        int(win.col_off * scale),
        int(win.row_off * scale),
        int(win.width * scale),
        int(win.height * scale),
    )


def rgb_composite(stack_4: np.ndarray) -> np.ndarray:
    r, g, b = stack_4[0], stack_4[1], stack_4[2]
    rgb = np.stack([r, g, b], axis=-1).astype(np.float64)
    out = np.zeros_like(rgb)
    for c in range(3):
        v = rgb[..., c]
        finite = np.isfinite(v)
        if not finite.any():
            continue
        lo, hi = np.percentile(v[finite], (2.0, 98.0))
        if hi <= lo:
            hi = lo + 1e-6
        out[..., c] = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32)


def discover_triplets(highres_root: Path, original_root: Path) -> List[ScenePaths]:
    bilinear_files = sorted(
        list(highres_root.glob("**/bilinear/**/*.tif"))
        + list(highres_root.glob("**/bilinear/**/*.tiff"))
    )
    triplets: List[ScenePaths] = []

    for bilinear in bilinear_files:
        try:
            rel_to_root = bilinear.relative_to(highres_root)
        except ValueError:
            continue

        parts = rel_to_root.parts
        if len(parts) < 3:
            continue

        site = parts[0]
        if parts[1].lower() != "bilinear":
            continue

        rel = Path(*parts[2:])
        bicubic = highres_root / site / "bicubic" / rel
        original = original_root / site / rel

        if bicubic.exists() and original.exists():
            triplets.append(
                ScenePaths(
                    site=site,
                    rel=rel,
                    original=original,
                    bilinear=bilinear,
                    bicubic=bicubic,
                )
            )

    return triplets


def iter_s2_triplets(highres_root: Path, original_root: Path) -> List[ScenePaths]:
    return [t for t in discover_triplets(highres_root, original_root) if t.rel.parts and t.rel.parts[0] == "S2"]


def load_sen2sr_model(project_root: Path, device: torch.device) -> torch.nn.Module:
    model_dir = project_root / "opensr_weights" / "SEN2SRLite_RGBN"
    hf_mlm = "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json"
    model_dir.mkdir(parents=True, exist_ok=True)
    mlm_path = model_dir / "mlm.json"

    if not mlm_path.exists():
        loader = mlstac.download(file=hf_mlm, output_dir=model_dir)
    else:
        loader = mlstac.load(mlm_path.as_posix())

    model = loader.compiled_model(device=device)
    model.eval()
    return model


def resize_stack_hw(
    stack: np.ndarray,
    out_h: int,
    out_w: int,
    *,
    mode: str,
) -> np.ndarray:
    """Resize (4, H, W) reflectance stack to (4, out_h, out_w). ``mode`` is ``nearest`` or ``bilinear``."""
    t = torch.from_numpy(stack.astype(np.float32, copy=False)).unsqueeze(0)
    if mode == "bilinear":
        t2 = F.interpolate(t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    else:
        t2 = F.interpolate(t, size=(out_h, out_w), mode="nearest")
    return t2.squeeze(0).numpy()


def center_crop_rgb(rgb: np.ndarray, side: int) -> np.ndarray:
    """Crop (H, W, 3) to ``side×side`` from the center."""
    h, w = rgb.shape[0], rgb.shape[1]
    side = min(side, h, w)
    cy, cx = h // 2, w // 2
    y0 = max(0, cy - side // 2)
    x0 = max(0, cx - side // 2)
    return rgb[y0 : y0 + side, x0 : x0 + side, :].copy()


def center_crop_rgb_at(rgb: np.ndarray, cy: int, cx: int, side: int) -> np.ndarray:
    """Crop ``side×side`` with integer center ``(cy, cx)`` clamped to fit ``rgb`` (H, W, 3)."""
    h, w = rgb.shape[0], rgb.shape[1]
    side = min(side, h, w)
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    return rgb[y0 : y0 + side, x0 : x0 + side, :].copy()


def otsu_threshold(img: np.ndarray, bins: int = 256) -> float:
    """Histogram Otsu threshold (finite values only)."""
    vals = img[np.isfinite(img)].ravel()
    if vals.size == 0:
        return 0.0

    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax <= vmin:
        return vmin

    hist, edges = np.histogram(vals, bins=bins, range=(vmin, vmax))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0

    w0 = np.cumsum(hist)
    w1 = np.cumsum(hist[::-1])[::-1]

    mu = np.cumsum(hist * centers)
    mu_t = mu[-1]

    eps = 1e-12
    mu0 = mu / (w0 + eps)
    mu1 = (mu_t - mu) / (w1 + eps)

    sigma_b2 = w0 * w1 * (mu0 - mu1) ** 2
    idx = int(np.nanargmax(sigma_b2))
    return float(centers[idx])


def ndwi_mcfeeters(stack_4: np.ndarray) -> np.ndarray:
    """McFeeters NDWI = (G - NIR) / (G + NIR); ``stack_4`` is (4, H, W) R,G,B,NIR."""
    g = stack_4[1].astype(np.float64)
    n = stack_4[3].astype(np.float64)
    return (g - n) / (g + n + 1e-8)


def _binary_erode_3x3(a: np.ndarray) -> np.ndarray:
    """Erode 2D bool with a 3×3 all-neighbors-True structuring element."""
    x = a.astype(bool, copy=False)
    out = x.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            rolled = np.roll(np.roll(x, dy, axis=0), dx, axis=1)
            out &= rolled
    return out


def shoreline_focus_nearest_to_point(
    stack_4: np.ndarray,
    pref_row: float,
    pref_col: float,
    *,
    min_water_frac: float = 0.02,
    max_water_frac: float = 0.98,
) -> Tuple[float | None, float | None]:
    """Shoreline pixel (row, col) nearest ``(pref_row, pref_col)`` using NDWI+Otsu edges."""
    ndwi = ndwi_mcfeeters(stack_4)
    if not np.any(np.isfinite(ndwi)):
        return None, None

    t = otsu_threshold(ndwi)
    water = (ndwi > t) & np.isfinite(ndwi)
    frac = float(np.mean(water))
    if frac < min_water_frac or frac > max_water_frac:
        return None, None

    finite = np.isfinite(ndwi)
    land = (~water) & finite

    ero = _binary_erode_3x3(water)
    shore_w = water & ~ero
    ero_l = _binary_erode_3x3(land)
    shore_l = land & ~ero_l
    shore = shore_w | shore_l
    if not np.any(shore):
        return None, None

    ys, xs = np.nonzero(shore)
    dr = ys.astype(np.float64) - pref_row
    dc = xs.astype(np.float64) - pref_col
    j = int(np.argmin(dr * dr + dc * dc))
    return float(ys[j]), float(xs[j])


def run_sen2sr_single_patch(
    lr_stack: np.ndarray, model: torch.nn.Module, device: torch.device
) -> np.ndarray:
    """One forward pass; ``lr_stack`` shape (4, H, W), typically 128×128."""
    x = torch.from_numpy(lr_stack).float().to(device)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    with torch.inference_mode():
        out = model(x.unsqueeze(0)).squeeze(0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    arr = out.detach().float().cpu().numpy()
    del out, x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return arr


def save_four_way_figure(
    scene: ScenePaths,
    project_root: Path,
    *,
    lr_patch: int = 128,
    interp_scale: int = 2,
    zoom2_center_fraction: float = 0.2,
    zoom2_focus: str = "shoreline",
    lr_shoreline_coarse_factor: int = 24,
    output_path: Path | None = None,
    device: torch.device | None = None,
    model: torch.nn.Module | None = None,
) -> Path:
    """Load crops, optional SR, write PNG. Returns path written."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if output_path is None:
        output_path = project_root / "opensr_outputs" / "four_way_original_bilinear_bicubic_sen2sr.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(scene.original) as src:
        h, w = src.height, src.width

    if zoom2_focus == "shoreline":
        win_lr = pick_lr_window_toward_shoreline(
            scene.original,
            lr_patch,
            coarse_factor=lr_shoreline_coarse_factor,
        )
    else:
        win_lr = center_lr_window(h, w, lr_patch)
    win_bi = scale_window(win_lr, interp_scale)
    win_bc = scale_window(win_lr, interp_scale)

    lr = read_rgbn_window(scene.original, win_lr)
    bi = read_rgbn_window(scene.bilinear, win_bi)
    bc = read_rgbn_window(scene.bicubic, win_bc)

    if lr.shape[1] != lr_patch or lr.shape[2] != lr_patch:
        pad_h = lr_patch - lr.shape[1]
        pad_w = lr_patch - lr.shape[2]
        lr = np.pad(lr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)

    own_model = model is None
    if own_model:
        model = load_sen2sr_model(project_root, device)

    try:
        sr = run_sen2sr_single_patch(lr, model, device)
    except Exception as exc:
        if own_model:
            del model
        raise RuntimeError(f"SEN2SR forward failed: {exc}") from exc

    if own_model:
        del model

    if sr.shape[1] != lr_patch * 4 or sr.shape[2] != lr_patch * 4:
        warnings.warn(
            f"SEN2SR output shape {sr.shape} is not 4× the LR patch ({lr_patch}); plotting anyway.",
            stacklevel=2,
        )

    th, tw = int(sr.shape[1]), int(sr.shape[2])
    # Same geographic footprint as SEN2SR (~2.5 m / px): LR 10 m → ×4; bilinear/bicubic 5 m → ×2.
    sr_scale_vs_lr = 4
    if th != lr_patch * sr_scale_vs_lr or tw != lr_patch * sr_scale_vs_lr:
        warnings.warn(
            f"SEN2SR size {(th, tw)} differs from 4× LR patch ({lr_patch}); "
            "alignment row uses SEN2SR dimensions as reference.",
            stacklevel=2,
        )

    titles_r1 = [
        f"Original (~10 m)\n{lr_patch}×{lr_patch} px",
        f"Bilinear (~5 m)\n×{interp_scale} interp",
        f"Bicubic (~5 m)\n×{interp_scale} interp",
        f"SEN2SR (~2.5 m)\n{th}×{tw} px",
    ]
    imgs_r1 = [rgb_composite(lr), rgb_composite(bi), rgb_composite(bc), rgb_composite(sr)]

    lr_on_sr_grid = resize_stack_hw(lr, th, tw, mode="nearest")
    bi_on_sr_grid = resize_stack_hw(bi, th, tw, mode="bilinear")
    bc_on_sr_grid = resize_stack_hw(bc, th, tw, mode="bilinear")

    zoom_side = max(32, int(min(th, tw) * zoom2_center_fraction))

    if zoom2_focus == "shoreline":
        cy_s, cx_s = shoreline_focus_nearest_to_point(sr, (th - 1) / 2.0, (tw - 1) / 2.0)
        if cy_s is None:
            cy_s, cx_s = float(th // 2), float(tw // 2)
            zoom_note = "geom center (no shoreline in SR patch)"
        else:
            zoom_note = "NDWI/Otsu shoreline (near patch center)"
    elif zoom2_focus == "center":
        cy_s, cx_s = float(th // 2), float(tw // 2)
        zoom_note = "geom center"
    else:
        raise ValueError("zoom2_focus must be 'shoreline' or 'center'")

    cy_i, cx_i = int(round(cy_s)), int(round(cx_s))

    titles_r2 = [
        f"Zoom ({zoom_note})\noriginal ↑×4 nearest",
        "bilinear ↑×2 (display)",
        "bicubic ↑×2 (display)",
        "SEN2SR (native)",
    ]
    imgs_r2 = [
        center_crop_rgb_at(rgb_composite(lr_on_sr_grid), cy_i, cx_i, zoom_side),
        center_crop_rgb_at(rgb_composite(bi_on_sr_grid), cy_i, cx_i, zoom_side),
        center_crop_rgb_at(rgb_composite(bc_on_sr_grid), cy_i, cx_i, zoom_side),
        center_crop_rgb_at(rgb_composite(sr), cy_i, cx_i, zoom_side),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    for c in range(4):
        axes[0, c].imshow(imgs_r1[c], extent=[0, 1, 0, 1], interpolation="nearest")
        axes[0, c].set_title(titles_r1[c], fontsize=9)
        axes[0, c].axis("off")
        axes[1, c].imshow(imgs_r2[c], extent=[0, 1, 0, 1], interpolation="nearest")
        axes[1, c].set_title(titles_r2[c], fontsize=8)
        axes[1, c].axis("off")

    fig.suptitle(f"{scene.site} | {scene.rel.as_posix()}", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save 4-way comparison (original, bilinear, bicubic, SEN2SR) for one S2 triplet."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Repo root (default: cwd).",
    )
    parser.add_argument(
        "--highres-root",
        type=Path,
        default=None,
        help="Upsampled tree with bilinear/bicubic (default: <project>/data/sat_images).",
    )
    parser.add_argument(
        "--original-root",
        type=Path,
        default=None,
        help="Original SDS sat_images (default: sibling SDS_performance_analysis/.../sat_images).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--pick",
        choices=("first", "random"),
        default="first",
        help="Which Sentinel-2 triplet to use.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed when --pick random.")
    parser.add_argument("--lr-patch", type=int, default=128, help="LR crop size (pixels).")
    parser.add_argument(
        "--interp-scale",
        type=int,
        default=2,
        help="Scale between original and bilinear/bicubic rasters (your convert_site_tiffs factor).",
    )
    parser.add_argument(
        "--zoom2-frac",
        type=float,
        default=0.2,
        help="Second row: crop side = this fraction of min(SEN2SR H, W). Smaller = tighter zoom.",
    )
    parser.add_argument(
        "--zoom2-focus",
        choices=("shoreline", "center"),
        default="shoreline",
        help="When 'shoreline': LR crop is chosen from full-scene NDWI/Otsu (coarse) near image center; "
        "row-2 zoom uses shoreline nearest patch center on SR. 'center' keeps geometric LR crop.",
    )
    parser.add_argument(
        "--lr-shoreline-coarse",
        type=int,
        default=24,
        help="Full-scene downsample factor for shoreline-based LR placement (larger = faster/coarser).",
    )
    args = parser.parse_args()

    project_root = (args.project_root or Path.cwd()).resolve()
    highres_root = args.highres_root or (project_root / "data" / "sat_images")
    original_root = args.original_root or (
        project_root.parent / "SDS_performance_analysis" / "data" / "sat_images"
    )

    triplets = iter_s2_triplets(highres_root, original_root)
    if not triplets:
        raise SystemExit(f"No S2 triplets under {highres_root} matching {original_root}.")

    if args.pick == "first":
        scene = triplets[0]
    else:
        rng = np.random.default_rng(args.seed)
        scene = triplets[int(rng.integers(0, len(triplets)))]

    out = save_four_way_figure(
        scene,
        project_root,
        lr_patch=args.lr_patch,
        interp_scale=args.interp_scale,
        zoom2_center_fraction=args.zoom2_frac,
        zoom2_focus=args.zoom2_focus,
        lr_shoreline_coarse_factor=args.lr_shoreline_coarse,
        output_path=args.output,
    )
    print("Wrote:", out.resolve())


if __name__ == "__main__":
    main()

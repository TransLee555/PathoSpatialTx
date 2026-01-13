#!/usr/bin/env python3
"""Stage 1 patch extraction with CLI controls."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import openslide

try:
    import opensdpc  # type: ignore
except ImportError:  # Optional dependency
    opensdpc = None

from histomicstk.saliency.tissue_detection import get_tissue_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract tissue patches from whole-slide images."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing input slide files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where mask and patch subfolders will be stored.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="Slide level used for patch extraction.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=512,
        help="Patch size in level-0 pixels.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=None,
        help="Stride in level-0 pixels. Defaults to the patch size.",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.2,
        help="Minimum tissue coverage ratio required to save a patch.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".svs"],
        help="Slide file extensions to process (e.g. .svs .sdpc).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for slides recursively.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip slides whose patch directory already exists.",
    )
    parser.add_argument(
        "--tissue-sigma",
        type=float,
        default=10.0,
        help="Gaussian sigma used by the tissue detector.",
    )
    parser.add_argument(
        "--tissue-min-area",
        type=int,
        default=1,
        help="Minimum connected-component size kept in the tissue mask.",
    )
    parser.add_argument(
        "--deconvolve-first",
        action="store_true",
        help="Apply color deconvolution before thresholding.",
    )
    parser.add_argument(
        "--mask-level",
        type=int,
        default=None,
        help="Override the slide level used to build the tissue mask (defaults to an automatically selected low-resolution level).",
    )
    parser.add_argument(
        "--mask-max-pixels",
        type=int,
        default=4_000_000,
        help="Maximum number of pixels allowed when automatically selecting the mask level.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def normalize_extensions(extensions: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    for ext in extensions:
        item = ext.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = f".{item}"
        normalized.append(item)
    return normalized


def discover_slides(root: Path, extensions: Sequence[str], recursive: bool) -> List[Path]:
    iterator = root.rglob("*") if recursive else root.iterdir()
    slides = [
        path for path in iterator if path.is_file() and path.suffix.lower() in extensions
    ]
    slides.sort()
    return slides


def open_slide_handle(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".sdpc":
        if opensdpc is None:
            raise RuntimeError("opensdpc is required to read SDPC files.")
        return opensdpc.OpenSdpc(str(path))
    return openslide.OpenSlide(str(path))


def select_mask_level(slide, requested_level: Optional[int], max_pixels: int) -> int:
    """Pick a slide level for mask generation based on a pixel budget."""
    max_index = slide.level_count - 1
    if requested_level is not None:
        return max(0, min(requested_level, max_index))
    for level in range(max_index, -1, -1):
        width, height = slide.level_dimensions[level]
        if width * height <= max_pixels:
            return level
    return max_index


def compute_tissue_mask(
    image_rgb: np.ndarray, args: argparse.Namespace
) -> np.ndarray:
    labeled, _ = get_tissue_mask(
        image_rgb,
        deconvolve_first=args.deconvolve_first,
        n_thresholding_steps=1,
        sigma=args.tissue_sigma,
        min_size=args.tissue_min_area,
    )
    return labeled > 0


def write_mask(mask: np.ndarray, mask_dir: Path, sample_name: str) -> None:
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / f"{sample_name}.jpg"
    cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))


def extract_patches_from_mask(
    slide,
    tissue_mask: np.ndarray,
    sample_name: str,
    mask_level: int,
    patch_level: int,
    patch_size_level0: int,
    patch_dir: Path,
    overlap_threshold: float,
    step_size_level0: Optional[int] = None,
) -> int:
    mask_downsample = float(slide.level_downsamples[mask_level])
    patch_downsample = float(slide.level_downsamples[patch_level])
    mask_patch_size = max(1, int(round(patch_size_level0 / mask_downsample)))
    mask_step = (
        mask_patch_size
        if step_size_level0 is None
        else max(1, int(round(step_size_level0 / mask_downsample)))
    )
    patch_level_size = max(1, int(round(patch_size_level0 / patch_downsample)))

    mask_height, mask_width = tissue_mask.shape
    if mask_height < mask_patch_size or mask_width < mask_patch_size:
        return 0

    patch_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for y in range(0, mask_height - mask_patch_size + 1, mask_step):
        for x in range(0, mask_width - mask_patch_size + 1, mask_step):
            window = tissue_mask[y : y + mask_patch_size, x : x + mask_patch_size]
            if window.mean() < overlap_threshold:
                continue

            level0_x = int(round(x * mask_downsample))
            level0_y = int(round(y * mask_downsample))
            region = slide.read_region(
                (level0_x, level0_y), patch_level, (patch_level_size, patch_level_size)
            )
            patch_rgb = np.array(region.convert("RGB"))
            filename = patch_dir / f"{sample_name}_{x}_{y}.jpg"
            cv2.imwrite(str(filename), patch_rgb)
            saved += 1

    return saved


def process_slide(slide_path: Path, args: argparse.Namespace) -> None:
    sample_name = slide_path.stem
    patch_dir = args.output_dir / "patches" / sample_name

    if args.skip_existing and patch_dir.exists() and any(patch_dir.iterdir()):
        logging.info("Skipping %s (patches already exist)", sample_name)
        return

    try:
        slide = open_slide_handle(slide_path)
    except Exception as exc:
        logging.error("Unable to open %s: %s", slide_path, exc)
        return

    try:
        patch_level = max(0, args.level)
        if patch_level >= slide.level_count:
            patch_level = slide.level_count - 1
            logging.warning("Adjusted patch level for %s to %d", sample_name, patch_level)

        mask_level = select_mask_level(slide, args.mask_level, args.mask_max_pixels)
        mask_dims = slide.level_dimensions[mask_level]
        logging.debug(
            "Processing %s with patch level %d (dims=%s) and mask level %d (dims=%s)",
            sample_name,
            patch_level,
            slide.level_dimensions[patch_level],
            mask_level,
            mask_dims,
        )

        region = slide.read_region((0, 0), mask_level, mask_dims)
        slide_rgb = np.array(region.convert("RGB"))

        tissue_mask = compute_tissue_mask(slide_rgb, args)
        write_mask(tissue_mask, args.output_dir / "mask", sample_name)

        saved = extract_patches_from_mask(
            slide=slide,
            tissue_mask=tissue_mask,
            sample_name=sample_name,
            mask_level=mask_level,
            patch_level=patch_level,
            patch_size_level0=args.patch_size,
            patch_dir=patch_dir,
            overlap_threshold=args.overlap_threshold,
            step_size_level0=args.step_size,
        )
        logging.info("Saved %d patches for %s", saved, sample_name)
    except Exception as exc:
        logging.error("Failed to process %s: %s", slide_path, exc)
    finally:
        slide.close()


def main() -> None:
    args = parse_args()
    args.root = args.root.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.extensions = normalize_extensions(args.extensions)

    if args.patch_size <= 0:
        raise ValueError("patch-size must be positive.")
    if args.step_size is not None and args.step_size <= 0:
        raise ValueError("step-size must be positive.")
    if args.mask_max_pixels <= 0:
        raise ValueError("mask-max-pixels must be positive.")

    if not args.root.exists():
        raise FileNotFoundError(f"Root directory {args.root} does not exist.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, args.log_level.upper())
    logging.basicConfig(level=log_level, format="%(asctime)s | %(levelname)s | %(message)s")

    slides = discover_slides(args.root, args.extensions, args.recursive)
    if not slides:
        logging.warning("No slides found under %s", args.root)
        return

    for slide_path in slides:
        process_slide(slide_path, args)


if __name__ == "__main__":
    main()


# python stage1_patch_extract.py --root /DATA/linzhiquan/lzq/jch_her2/wsi/ --output-dir /DATA/linzhiquan/lzq/PathoSpatialTx_GITHUB/ --patch-size 512 --level 1 --mask-max-pixels 2000000 --overlap-threshold 0.2 --skip-existing

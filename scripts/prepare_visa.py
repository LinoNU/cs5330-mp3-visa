"""Build the curated VisA subset used in CS 5330 Mini-Project 3.

The script reads the official VisA release (Zou et al., ECCV 2022), selects a
fixed set of subsets, resplits each one into reference / dev / test sets,
resizes images and masks, binarizes the masks, strips image metadata, and
writes a manifest. Every step is deterministic: running the script twice with
the same arguments produces identical files.

Expected layout of the extracted release (``--raw-dir``)::

    <raw-dir>/
        cashew/
            Data/
                Images/
                    Normal/    000.JPG, 001.JPG, ...
                    Anomaly/   000.JPG, 001.JPG, ...
                Masks/
                    Anomaly/   000.png, 001.png, ...
            image_anno.csv
        macaroni1/
        ...

Example::

    python scripts/prepare_visa.py --raw-dir raw/VisA --out-dir data --overwrite
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

LOGGER = logging.getLogger("prepare_visa")

CORE_SUBSETS: tuple[str, ...] = ("cashew", "macaroni1")
EXTENSION_SUBSETS: tuple[str, ...] = ("chewinggum", "fryum", "pipe_fryum", "macaroni2")
DEFAULT_SUBSETS: tuple[str, ...] = CORE_SUBSETS + EXTENSION_SUBSETS

# Number of images per split and label. Reference images are normal only:
# students draw their few-shot memory bank from this pool.
SPLIT_SPEC: dict[str, dict[str, int]] = {
    "reference": {"normal": 40, "anomaly": 0},
    "dev": {"normal": 20, "anomaly": 20},
    "test": {"normal": 50, "anomaly": 50},
}

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
JPEG_QUALITY = 90
MASK_ON = 255

MANIFEST_COLUMNS: list[str] = [
    "subset",
    "split",
    "label",
    "image_path",
    "mask_path",
    "width",
    "height",
    "defect_area_px",
    "original_filename",
]


class ValidationError(RuntimeError):
    """Raised when the curated output fails a consistency check."""


@dataclass(frozen=True)
class SourceImage:
    """One image from the original release, with its mask if anomalous."""

    subset: str
    label: str  # "normal" or "anomaly"
    image_path: Path
    mask_path: Path | None
    original_name: str  # path relative to the release root, for traceability


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _list_images(folder: Path) -> list[Path]:
    """Return image files in ``folder``, sorted by name for determinism."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Expected folder not found: {folder}")
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def discover_subset(raw_dir: Path, subset: str) -> list[SourceImage]:
    """Find all normal and anomalous images (and anomaly masks) for one subset."""
    data_dir = raw_dir / subset / "Data"
    normal_images = _list_images(data_dir / "Images" / "Normal")
    anomaly_images = _list_images(data_dir / "Images" / "Anomaly")
    mask_dir = data_dir / "Masks" / "Anomaly"

    masks_by_stem = {p.stem: p for p in _list_images(mask_dir)}

    def original_name(path: Path) -> str:
        return path.relative_to(raw_dir).as_posix()

    sources = [
        SourceImage(subset, "normal", p, None, original_name(p)) for p in normal_images
    ]
    missing_masks: list[str] = []
    for image_path in anomaly_images:
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is None:
            missing_masks.append(image_path.name)
            continue
        sources.append(
            SourceImage(subset, "anomaly", image_path, mask_path, original_name(image_path))
        )

    if missing_masks:
        raise ValidationError(
            f"{subset}: {len(missing_masks)} anomalous images have no mask, "
            f"e.g. {missing_masks[:5]}"
        )

    LOGGER.info(
        "%s: found %d normal and %d anomalous images",
        subset,
        len(normal_images),
        len(anomaly_images),
    )
    _cross_check_annotation_csv(raw_dir / subset, len(normal_images), len(anomaly_images))
    return sources


def _cross_check_annotation_csv(subset_dir: Path, n_normal: int, n_anomaly: int) -> None:
    """Compare folder counts with the release's image_anno.csv, if present."""
    csv_path = subset_dir / "image_anno.csv"
    if not csv_path.is_file():
        LOGGER.warning("%s: no image_anno.csv found; skipping cross-check", subset_dir.name)
        return
    annotations = pd.read_csv(csv_path)
    if "label" not in annotations.columns:
        LOGGER.warning("%s: image_anno.csv has no 'label' column", subset_dir.name)
        return
    # Anomalous rows carry defect-type names (e.g. "burnt", "small scratches"),
    # so anything that is not "normal" counts as anomalous.
    is_normal = annotations["label"].astype(str).str.strip().str.lower() == "normal"
    csv_normal = int(is_normal.sum())
    csv_anomaly = int((~is_normal).sum())
    if (csv_normal, csv_anomaly) != (n_normal, n_anomaly):
        raise ValidationError(
            f"{subset_dir.name}: folders contain {n_normal} normal / {n_anomaly} anomalous "
            f"images but image_anno.csv lists {csv_normal} / {csv_anomaly}"
        )


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_subset(
    sources: list[SourceImage], subset: str, seed: int
) -> dict[str, list[SourceImage]]:
    """Assign images to splits by sampling without replacement.

    The random generator is seeded from both ``seed`` and the subset name, so
    adding or removing a subset never changes the splits of the others.
    """
    rng = np.random.default_rng([seed, zlib.crc32(subset.encode("utf-8"))])
    splits: dict[str, list[SourceImage]] = {name: [] for name in SPLIT_SPEC}

    for label in ("normal", "anomaly"):
        pool = [s for s in sources if s.label == label]
        needed = sum(spec[label] for spec in SPLIT_SPEC.values())
        if len(pool) < needed:
            raise ValidationError(
                f"{subset}: need {needed} {label} images but only {len(pool)} are available"
            )
        order = rng.permutation(len(pool))
        cursor = 0
        for split_name, spec in SPLIT_SPEC.items():
            count = spec[label]
            splits[split_name].extend(pool[i] for i in order[cursor : cursor + count])
            cursor += count
    return splits


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------


def target_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Scale (width, height) so the longer side equals ``long_side``.

    Images that are already smaller are left at their original size.
    """
    scale = long_side / max(width, height)
    if scale >= 1.0:
        return width, height
    return max(1, round(width * scale)), max(1, round(height * scale))


def load_binary_mask(mask_path: Path) -> np.ndarray:
    """Load a VisA mask and return a boolean array (True = anomaly).

    Original masks may be multi-class indexed, greyscale, or RGB; any
    non-zero value is treated as anomalous.
    """
    with Image.open(mask_path) as mask:
        values = np.asarray(mask)
    if values.ndim == 3:
        return values.any(axis=2)
    return values > 0


def process_image(
    source: SourceImage, image_out: Path, mask_out: Path | None, long_side: int
) -> tuple[int, int, int | None]:
    """Resize and save one image (and its mask). Returns (width, height, defect area)."""
    with Image.open(source.image_path) as original:
        image = original.convert("RGB")
    new_size = target_size(image.width, image.height, long_side)
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    # Saving a freshly converted image without an `exif` argument writes no metadata.
    resized.save(image_out, format="JPEG", quality=JPEG_QUALITY, optimize=True)

    if source.mask_path is None or mask_out is None:
        return new_size[0], new_size[1], None

    mask = load_binary_mask(source.mask_path)
    if mask.shape != (image.height, image.width):
        raise ValidationError(
            f"{source.image_path}: mask shape {mask.shape} does not match "
            f"image size {(image.height, image.width)}"
        )
    mask_image = Image.fromarray(np.where(mask, MASK_ON, 0).astype(np.uint8))
    mask_resized = mask_image.resize(new_size, Image.Resampling.NEAREST)
    mask_resized.save(mask_out, format="PNG", optimize=True)
    defect_area = int((np.asarray(mask_resized) == MASK_ON).sum())
    return new_size[0], new_size[1], defect_area


def write_subset(
    splits: dict[str, list[SourceImage]],
    subset: str,
    out_dir: Path,
    path_root: Path,
    long_side: int,
) -> list[dict[str, object]]:
    """Write all images for one subset and return its manifest rows."""
    rows: list[dict[str, object]] = []
    for split_name, sources in splits.items():
        split_dir = out_dir / subset / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        counters = {"normal": 0, "anomaly": 0}
        # Sort within the split so file indices do not depend on sampling order.
        for source in sorted(sources, key=lambda s: (s.label, s.image_path.name)):
            index = counters[source.label]
            counters[source.label] += 1
            stem = f"{subset}_{split_name}_{source.label}_{index:03d}"
            image_out = split_dir / f"{stem}.jpg"
            mask_out = split_dir / f"{stem}_mask.png" if source.mask_path else None
            width, height, defect_area = process_image(source, image_out, mask_out, long_side)
            rows.append(
                {
                    "subset": subset,
                    "split": split_name,
                    "label": source.label,
                    "image_path": image_out.relative_to(path_root).as_posix(),
                    "mask_path": mask_out.relative_to(path_root).as_posix() if mask_out else "",
                    "width": width,
                    "height": height,
                    "defect_area_px": defect_area if defect_area is not None else "",
                    "original_filename": source.original_name,
                }
            )
    LOGGER.info("%s: wrote %d images", subset, len(rows))
    return rows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_output(manifest: pd.DataFrame, path_root: Path) -> None:
    """Check the curated dataset and raise ValidationError on any problem."""
    errors: list[str] = []

    # Split counts.
    for (subset, split_name, label), group in manifest.groupby(["subset", "split", "label"]):
        expected = SPLIT_SPEC[split_name][label]
        if len(group) != expected:
            errors.append(f"{subset}/{split_name}/{label}: {len(group)} images, expected {expected}")

    # No source image may appear twice.
    duplicates = manifest[manifest.duplicated(["subset", "original_filename"], keep=False)]
    if not duplicates.empty:
        errors.append(
            f"{len(duplicates)} rows reuse a source image across splits, e.g. "
            f"{duplicates['original_filename'].head(3).tolist()}"
        )

    # Image / mask consistency.
    for row in manifest.itertuples(index=False):
        image_path = path_root / row.image_path
        if not image_path.is_file():
            errors.append(f"missing image file {row.image_path}")
            continue
        with Image.open(image_path) as image:
            if image.getexif():
                errors.append(f"{row.image_path}: metadata was not stripped")
            image_size = image.size

        if row.label == "normal":
            if row.mask_path:
                errors.append(f"{row.image_path}: normal image has a mask")
            continue

        if not row.mask_path:
            errors.append(f"{row.image_path}: anomalous image has no mask")
            continue
        with Image.open(path_root / row.mask_path) as mask:
            mask_size = mask.size
            values = np.unique(np.asarray(mask))
        if mask_size != image_size:
            errors.append(f"{row.mask_path}: size {mask_size} differs from image {image_size}")
        if not set(values.tolist()) <= {0, MASK_ON}:
            errors.append(f"{row.mask_path}: values {values.tolist()} are not binary")
        if MASK_ON not in values:
            errors.append(
                f"{row.mask_path}: defect vanished after resizing; try a larger --size"
            )

    if errors:
        for message in errors:
            LOGGER.error(message)
        raise ValidationError(f"{len(errors)} validation check(s) failed")
    LOGGER.info("All validation checks passed")


def report_small_defects(manifest: pd.DataFrame, min_defect_px: int) -> pd.DataFrame:
    """Log anomalous images whose defect covers fewer than ``min_defect_px`` pixels."""
    anomalies = manifest[manifest["label"] == "anomaly"].copy()
    anomalies["defect_area_px"] = pd.to_numeric(anomalies["defect_area_px"])
    small = anomalies[anomalies["defect_area_px"] < min_defect_px]
    if small.empty:
        LOGGER.info("No defects smaller than %d px after resizing", min_defect_px)
    else:
        LOGGER.warning(
            "%d anomalous image(s) have defects smaller than %d px after resizing:",
            len(small),
            min_defect_px,
        )
        for row in small.itertuples(index=False):
            LOGGER.warning("  %s (%d px)", row.image_path, row.defect_area_px)
    return small


def log_summary(manifest: pd.DataFrame) -> None:
    """Log image counts per subset, split, and label."""
    table = (
        manifest.pivot_table(
            index="subset", columns=["split", "label"], values="image_path", aggfunc="count"
        )
        .fillna(0)
        .astype(int)
    )
    LOGGER.info("Image counts:\n%s", table.to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--raw-dir", type=Path, required=True, help="Folder containing the extracted VisA subsets."
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data"), help="Output folder (default: data)."
    )
    parser.add_argument(
        "--size", type=int, default=512, help="Length of the longer image side in px (default: 512)."
    )
    parser.add_argument("--seed", type=int, default=5330, help="Random seed (default: 5330).")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=list(DEFAULT_SUBSETS),
        help="Subsets to process (default: core and extension subsets).",
    )
    parser.add_argument(
        "--min-defect-px",
        type=int,
        default=50,
        help="Warn about defects smaller than this many pixels after resizing (default: 50).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Delete existing subset folders before writing."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    raw_dir: Path = args.raw_dir.resolve()
    out_dir: Path = args.out_dir.resolve()
    # Manifest paths are relative to the folder that contains out_dir (the repo root).
    path_root = out_dir.parent

    for subset in args.subsets:
        subset_out = out_dir / subset
        if subset_out.exists():
            if not args.overwrite:
                LOGGER.error("%s already exists; rerun with --overwrite to replace it", subset_out)
                return 1
            shutil.rmtree(subset_out)

    try:
        rows: list[dict[str, object]] = []
        for subset in args.subsets:
            sources = discover_subset(raw_dir, subset)
            splits = split_subset(sources, subset, args.seed)
            rows.extend(write_subset(splits, subset, out_dir, path_root, args.size))

        manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS).sort_values(
            ["subset", "split", "label", "image_path"], ignore_index=True
        )
        manifest_path = out_dir / "manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        LOGGER.info("Wrote %s (%d rows)", manifest_path, len(manifest))

        validate_output(manifest, path_root)
        report_small_defects(manifest, args.min_defect_px)
        log_summary(manifest)
    except (ValidationError, FileNotFoundError) as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Flatten and merge downloaded Kaggle datasets into a unified Classification/ tree.

Input (from download_data.py):
    data/raw/
        the_wildfire_dataset/{train,val,test}/{fire,nofire}/{subcategory}/
        Classification/{train,valid,test}/{fire,nofire}/   (FlameVision)

Output:
    data/processed/Classification/{train,val,test}/{fire,nofire}/

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --raw data/raw --out data/processed/Classification
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
from pathlib import Path


FIRE_SUBCATEGORIES = ["Both_smoke_and_fire", "Smoke_from_fires"]
NOFIRE_SUBCATEGORIES = [
    "Fire_confounding_elements",
    "Forested_areas_without_confounding_elements",
    "Smoke_confounding_elements",
]


def _move_all(src: Path, dest: Path) -> None:
    if not src.is_dir():
        return
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file():
            target = dest / item.name
            if target.exists():
                stem, suffix = item.stem, item.suffix
                target = dest / f"{stem}_{random.randint(10000, 99999)}{suffix}"
            shutil.move(str(item), str(target))


def flatten_wildfire_dataset(raw_dir: Path, out_dir: Path) -> None:
    """Collapses subcategory dirs from 'the_wildfire_dataset' into fire/nofire."""
    wf_root = raw_dir / "the_wildfire_dataset"
    if not wf_root.is_dir():
        wf_root = raw_dir / "Classification1"
    if not wf_root.is_dir():
        print(f"Wildfire dataset not found at {wf_root}; skipping.")
        return

    split_map = {"train": "train", "val": "val", "test": "test"}
    for src_split, dst_split in split_map.items():
        for subcat in FIRE_SUBCATEGORIES:
            src = wf_root / src_split / "fire" / subcat
            _move_all(src, out_dir / dst_split / "fire")
        remaining_fire = wf_root / src_split / "fire"
        _move_all(remaining_fire, out_dir / dst_split / "fire")

        for subcat in NOFIRE_SUBCATEGORIES:
            src = wf_root / src_split / "nofire" / subcat
            _move_all(src, out_dir / dst_split / "nofire")
        remaining_nofire = wf_root / src_split / "nofire"
        _move_all(remaining_nofire, out_dir / dst_split / "nofire")

    print(f"Wildfire dataset flattened → {out_dir}")


def merge_flamevision(raw_dir: Path, out_dir: Path) -> None:
    """Merges the FlameVision Classification/ into the unified output tree."""
    fv_root = raw_dir / "Classification"
    if not fv_root.is_dir():
        print(f"FlameVision data not found at {fv_root}; skipping.")
        return

    split_map = {"train": "train", "valid": "val", "test": "test"}
    for src_split, dst_split in split_map.items():
        for cls in ("fire", "nofire"):
            src = fv_root / src_split / cls
            _move_all(src, out_dir / dst_split / cls)

    print(f"FlameVision merged → {out_dir}")


def rename_images_randomly(directory: Path) -> None:
    """Renames images with random numeric names to avoid filename collisions."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    for p in list(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            new_name = directory / f"{random.randint(1, 10_000_000)}{p.suffix}"
            while new_name.exists():
                new_name = directory / f"{random.randint(1, 10_000_000)}{p.suffix}"
            p.rename(new_name)


def check_for_duplicates(out_dir: Path) -> None:
    """Prints duplicate image counts across splits (by MD5 hash)."""

    def get_hashes(d: Path) -> dict[str, str]:
        result = {}
        if not d.is_dir():
            return result
        for p in d.rglob("*"):
            if p.is_file():
                h = hashlib.md5(p.read_bytes()).hexdigest()
                result[str(p)] = h
        return result

    splits = ["train", "val", "test"]
    hashes = {s: get_hashes(out_dir / s) for s in splits}

    train_vals = set(hashes["train"].values())
    val_vals = set(hashes["val"].values())
    test_vals = set(hashes["test"].values())

    print(f"Duplicate images train↔val: {len(train_vals & val_vals)}")
    print(f"Duplicate images train↔test: {len(train_vals & test_vals)}")
    print(f"Duplicate images val↔test: {len(val_vals & test_vals)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare wildfire detection dataset")
    parser.add_argument("--raw", default="data/raw", help="Raw data directory")
    parser.add_argument(
        "--out",
        default="data/processed/Classification",
        help="Output directory",
    )
    parser.add_argument(
        "--skip-rename",
        action="store_true",
        help="Skip random image renaming step",
    )
    args = parser.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    flatten_wildfire_dataset(raw, out)
    merge_flamevision(raw, out)

    if not args.skip_rename:
        for split in ("val", "test"):
            for cls in ("fire", "nofire"):
                d = out / split / cls
                if d.is_dir():
                    rename_images_randomly(d)

    check_for_duplicates(out)

    total = sum(
        len(list((out / split / cls).iterdir()))
        for split in ("train", "val", "test")
        for cls in ("fire", "nofire")
        if (out / split / cls).is_dir()
    )
    print(f"\nDataset ready at {out}  (total images: {total})")


if __name__ == "__main__":
    main()

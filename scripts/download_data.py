"""
Download Kaggle datasets for WildFireVision.

Requires Kaggle credentials via ~/.kaggle/kaggle.json or env vars
KAGGLE_USERNAME and KAGGLE_KEY.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --dest data/raw
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASETS = [
    ("elmadafri/the-wildfire-dataset", "the-wildfire-dataset"),
    ("anamibnjafar0/flamevision", "flamevision"),
]


def download_dataset(slug: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {slug} → {dest}/")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            slug,
            "-p",
            str(dest),
            "--unzip",
        ],
        check=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed for {slug}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Kaggle wildfire datasets")
    parser.add_argument("--dest", default="data/raw", help="Destination directory")
    args = parser.parse_args()
    dest = Path(args.dest)

    for slug, _ in DATASETS:
        download_dataset(slug, dest)

    print("\nAll datasets downloaded. Run `python scripts/prepare_data.py` next.")


if __name__ == "__main__":
    main()

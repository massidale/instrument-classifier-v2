#!/usr/bin/env python
"""Download and extract the IRMAS dataset (training + testing) from Zenodo.

Zenodo record 1290750. Training clips are single-instrument 3s excerpts;
testing clips are 5-20s polyphonic excerpts with multi-label .txt annotations.
The three testing parts are consolidated into a single ``IRMAS-TestingData/``.

If you already have IRMAS-TrainingData elsewhere (e.g. the old project), pass
``--link-train /path/to/IRMAS-TrainingData`` to symlink it instead of downloading.
"""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://zenodo.org/record/1290750/files/"
TRAIN_ZIP = "IRMAS-TrainingData.zip"
TEST_ZIPS = [
    "IRMAS-TestingData-Part1.zip",
    "IRMAS-TestingData-Part2.zip",
    "IRMAS-TestingData-Part3.zip",
]


def _progress(block_num, block_size, total_size):
    done = block_num * block_size
    if total_size > 0:
        pct = min(100.0, done * 100.0 / total_size)
        print(f"\r  {pct:5.1f}%  ({done // (1024*1024)} MB)", end="", flush=True)


def _download(name: str, dest: Path) -> Path:
    zip_path = dest / name
    if zip_path.exists():
        print(f"  cached: {name}")
        return zip_path
    print(f"Downloading {name}")
    urllib.request.urlretrieve(BASE + name + "?download=1", zip_path, _progress)
    print()
    return zip_path


def _extract(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--link-train", type=Path, default=None,
                        help="Symlink an existing IRMAS-TrainingData folder instead of downloading")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    tmp = data_dir / "_zips"
    tmp.mkdir(parents=True, exist_ok=True)

    # --- Training data ---
    train_dir = data_dir / "IRMAS-TrainingData"
    if args.link_train:
        if not train_dir.exists():
            train_dir.symlink_to(args.link_train.resolve(), target_is_directory=True)
            print(f"Linked training data: {train_dir} -> {args.link_train}")
    elif not args.skip_train and not train_dir.exists():
        _extract(_download(TRAIN_ZIP, tmp), data_dir)

    # --- Testing data: download all parts, consolidate into one folder ---
    test_dir = data_dir / "IRMAS-TestingData"
    if not args.skip_test:
        test_dir.mkdir(parents=True, exist_ok=True)
        for name in TEST_ZIPS:
            _extract(_download(name, tmp), test_dir)
        wavs = list(test_dir.rglob("*.wav"))
        print(f"Testing clips available: {len(wavs)}")

    print("Done.")


if __name__ == "__main__":
    main()

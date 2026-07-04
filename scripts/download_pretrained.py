#!/usr/bin/env python
"""Download the AudioSet-pretrained PANNs CNN14 checkpoint (Cnn14_mAP=0.431.pth).

Source: Kong et al. 2020, hosted on Zenodo (record 3987831).
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
DEFAULT_OUT = Path("checkpoints/pretrained/Cnn14_mAP=0.431.pth")


def _progress(block_num, block_size, total_size):
    done = block_num * block_size
    if total_size > 0:
        pct = min(100.0, done * 100.0 / total_size)
        print(f"\r  {pct:5.1f}%  ({done // (1024*1024)} MB)", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if args.out.exists():
        print(f"Already present: {args.out}")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading CNN14 checkpoint -> {args.out}")
    urllib.request.urlretrieve(URL, args.out, _progress)
    print("\nDone.")


if __name__ == "__main__":
    main()

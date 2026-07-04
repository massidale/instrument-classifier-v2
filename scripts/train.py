#!/usr/bin/env python
"""CLI entrypoint: finetune CNN14 on IRMAS. Thin wrapper over the package."""

from instrument_classifier.train import main

if __name__ == "__main__":
    main()

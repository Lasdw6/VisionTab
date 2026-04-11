#!/usr/bin/env python3
"""
Stage 1 — text-only Fill-in-the-Middle (FIM) QLoRA fine-tuning on the base model.

Training is implemented as a Jupyter notebook (Kaggle / local GPU) so dependency
versions match the Gemma 4 + bitsandbytes stack. This script prints the path and
optional opens documentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tab model — text FIM fine-tuning (see notebook for full run)"
    )
    parser.add_argument(
        "--print-path",
        action="store_true",
        help="Print absolute path to the training notebook and exit",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    nb = repo / "notebooks" / "text_fim_training.ipynb"
    if args.print_path:
        print(nb)
        return

    print("Text FIM training for the Tab model is defined in:")
    print(f"  {nb}")
    print()
    print("Typical setup: Jupyter on Kaggle with GPU T4 x2, or a local CUDA machine")
    print("with PyTorch, transformers, peft, and bitsandbytes per that notebook.")


if __name__ == "__main__":
    main()

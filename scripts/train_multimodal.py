#!/usr/bin/env python3
"""
Stage 2 — multimodal (screenshot + FIM text) fine-tuning on top of the text FIM adapter.

Implemented as a notebook; run after Stage 1 and Stage 0 multimodal data prep.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tab model — multimodal fine-tuning (see notebook for full run)"
    )
    parser.add_argument(
        "--print-path",
        action="store_true",
        help="Print absolute path to the training notebook and exit",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    nb = repo / "notebooks" / "multimodal_training.ipynb"
    if args.print_path:
        print(nb)
        return

    print("Multimodal training for the Tab model is defined in:")
    print(f"  {nb}")
    print()
    print("Prerequisites: FIM dataset, multimodal dataset from scripts/prepare_data.py,")
    print("and a text FIM adapter from Stage 1 (text_fim_training.ipynb).")


if __name__ == "__main__":
    main()

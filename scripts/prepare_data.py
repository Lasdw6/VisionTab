#!/usr/bin/env python3
"""
Stage 0 — prepare FIM and/or multimodal datasets for the Tab model.

Mirrors the role of OpenComposer's ``prepare_data.py`` for this project's pipeline.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Tab model datasets (text FIM and/or multimodal)"
    )
    parser.add_argument(
        "--stage",
        choices=("fim", "multimodal", "all"),
        default="all",
        help="fim: text FIM only; multimodal: vision+text pairs (needs FIM output); all: both",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config YAML (relative to repo root)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override max multimodal samples (multimodal / all stages only)",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = (repo / args.config).resolve()
    if not cfg.is_file():
        print(f"Config not found: {cfg}", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    pypath = str(repo)
    env["PYTHONPATH"] = pypath + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else pypath

    def run_mod(module: str, extra: list[str] | None = None) -> int:
        cmd = [sys.executable, "-m", module, "--config", str(cfg)]
        if extra:
            cmd.extend(extra)
        return subprocess.call(cmd, cwd=str(repo), env=env)

    if args.stage in ("fim", "all"):
        code = run_mod("tab.prepare_dataset")
        if code != 0:
            sys.exit(code)

    if args.stage in ("multimodal", "all"):
        extra = []
        if args.max_samples is not None:
            extra.extend(["--max-samples", str(args.max_samples)])
        sys.exit(run_mod("tab.multimodal_dataset", extra))


if __name__ == "__main__":
    main()

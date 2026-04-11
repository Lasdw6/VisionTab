#!/usr/bin/env python3
"""
Stage 3 — inference and smoke evaluation for the Tab (VisionTab) multimodal model.

Delegates to the repo-root ``run_multimodal_inference.py`` so CLI flags stay in one place.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "run_multimodal_inference.py"
    cmd = [sys.executable, str(script), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=str(repo)))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Unified export entry point for Pi3 / DA3 model venvs.

Called by the pipeline (run_scene.py) or directly:

    vendor/Pi3/.venv/bin/python scripts/export.py pi3 --image_dir ... --output ...
    vendor/Depth-Anything-3/.venv/bin/python scripts/export.py da3 --image_dir ... --output ...
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make colmap_priors importable even when running under a model venv.
_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from colmap_priors.export_poses import main_da3, main_pi3  # type: ignore  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("pi3", "da3"):
        print("Usage: export.py {pi3|da3} [model args...]", file=sys.stderr)
        raise SystemExit(1)

    model = sys.argv[1]
    remaining = sys.argv[2:]

    if model == "pi3":
        raise SystemExit(main_pi3(remaining))
    else:
        raise SystemExit(main_da3(remaining))

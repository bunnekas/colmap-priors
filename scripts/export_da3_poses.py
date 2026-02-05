#!/usr/bin/env python3
import sys
from pathlib import Path

# Make colmap_priors importable even inside the DA3 venv
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from colmap_priors.export_poses import main_da3 as main  # type: ignore  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())

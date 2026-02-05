from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs into os.environ.

    This is intentionally minimal (no export semantics, no variable expansion) and
    matches the lightweight env files used by this repo.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Env file not found: {path}")

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip().strip("'").strip('"')

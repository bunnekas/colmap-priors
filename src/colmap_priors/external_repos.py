from __future__ import annotations

import os
import sys
from pathlib import Path


def add_repo_to_syspath(env_var: str) -> None:
    """Prepend an external repo path (from an env var) to sys.path.

    This is used by the Pi3/DA3 exporters that live in *other* repos/venvs.
    It keeps this package importable while letting the model code be imported
    from its own checkout.
    """

    repo = os.environ.get(env_var, "")
    if not repo:
        return

    p = str(Path(repo).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)

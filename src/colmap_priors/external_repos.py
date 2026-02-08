from __future__ import annotations

import sys
from pathlib import Path

# Repo root (two levels up from this file: src/colmap_priors/ -> repo root)
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Vendored submodule directories.
VENDOR_PI3 = _REPO_ROOT / "vendor" / "Pi3"
VENDOR_DA3 = _REPO_ROOT / "vendor" / "Depth-Anything-3"

_VENDOR: dict[str, Path] = {
    "PI3": VENDOR_PI3,
    "DA3": VENDOR_DA3,
}


def add_vendor_to_syspath(model: str) -> None:
    """Prepend the vendored submodule for *model* (``"PI3"`` or ``"DA3"``) to ``sys.path``."""
    vendor_dir = _VENDOR.get(model)
    if vendor_dir is None:
        raise ValueError(f"Unknown vendor model {model!r}; expected one of {sorted(_VENDOR)}")
    if not vendor_dir.is_dir():
        raise FileNotFoundError(
            f"Vendor submodule not found at {vendor_dir}. "
            "Run `git submodule update --init --recursive` (or `just vendor`)."
        )
    p = str(vendor_dir.resolve())
    if p not in sys.path:
        sys.path.insert(0, p)

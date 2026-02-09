from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .external_repos import VENDOR_DA3, VENDOR_PI3


@dataclass
class PipelineConfig:
    """Typed pipeline configuration loaded from a YAML file."""

    # Paths
    data_root: Path
    exp_root: Path
    prior_position_std: float

    colmap_exe: str = "colmap"
    ref_model_dir: Path | None = None
    max_images: int = 0

    # Pipeline toggles
    run_baseline: bool = False
    run_pi3: bool = False
    run_da3: bool = True
    run_plot: bool = True

    # Model interpreters (default: vendor venvs)
    pi3_python: str = field(default="")
    da3_python: str = field(default="")

    # Pi3 knobs
    pi3_interval: int = 1

    # DA3 knobs
    da3_window: int = 20
    da3_overlap: int = 10
    da3_autocast: bool = True

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.exp_root = Path(self.exp_root)
        if self.ref_model_dir is not None:
            self.ref_model_dir = Path(self.ref_model_dir)
        if not self.pi3_python:
            self.pi3_python = str(VENDOR_PI3 / ".venv" / "bin" / "python")
        if not self.da3_python:
            self.da3_python = str(VENDOR_DA3 / ".venv" / "bin" / "python")


def load_config(path: Path) -> PipelineConfig:
    """Load a YAML config file into a :class:`PipelineConfig`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return PipelineConfig(**raw)

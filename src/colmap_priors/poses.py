from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PoseW2C:
    qvec: np.ndarray  # (4,) qw,qx,qy,qz
    tvec: np.ndarray  # (3,)


@dataclass(frozen=True)
class PoseSet:
    poses: dict[str, PoseW2C]

    def __len__(self) -> int:
        return len(self.poses)

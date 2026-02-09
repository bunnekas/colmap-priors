from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Pose data classes (formerly poses.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseW2C:
    qvec: np.ndarray  # (4,) qw,qx,qy,qz
    tvec: np.ndarray  # (3,)


@dataclass(frozen=True)
class PoseSet:
    poses: dict[str, PoseW2C]

    def __len__(self) -> int:
        return len(self.poses)


# ---------------------------------------------------------------------------
# Quaternion / rotation conversions
# ---------------------------------------------------------------------------


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    q = np.asarray(qvec, dtype=np.float64).reshape(4)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    K = (
        np.array(
            [
                [R[0, 0] - R[1, 1] - R[2, 2], R[1, 0] + R[0, 1], R[2, 0] + R[0, 2], R[1, 2] - R[2, 1]],
                [R[1, 0] + R[0, 1], R[1, 1] - R[0, 0] - R[2, 2], R[2, 1] + R[1, 2], R[2, 0] - R[0, 2]],
                [R[2, 0] + R[0, 2], R[2, 1] + R[1, 2], R[2, 2] - R[0, 0] - R[1, 1], R[0, 1] - R[1, 0]],
                [R[1, 2] - R[2, 1], R[2, 0] - R[0, 2], R[0, 1] - R[1, 0], R[0, 0] + R[1, 1] + R[2, 2]],
            ],
            dtype=np.float64,
        )
        / 3.0
    )
    w, v = np.linalg.eigh(K)
    q = v[:, np.argmax(w)]
    qvec = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def c2w_to_w2c_and_center(T_c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = np.asarray(T_c2w, dtype=np.float64).reshape(4, 4)
    R_c2w = T[:3, :3]
    t_c2w = T[:3, 3]
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    C = t_c2w.copy()
    return R_w2c, t_w2c, C

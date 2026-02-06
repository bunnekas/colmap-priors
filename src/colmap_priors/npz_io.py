from __future__ import annotations

from pathlib import Path

import numpy as np


def _basename_list(paths: object) -> list[str]:
    arr = np.asarray(paths, dtype=object).reshape(-1)
    return [Path(str(p)).name for p in arr.tolist()]


def load_npz_order(npz_path: Path, *, key: str = "image_paths") -> list[str]:
    """Return basename ordering from an NPZ (default key: 'image_paths')."""

    z = np.load(npz_path, allow_pickle=True)
    if key not in z:
        raise KeyError(f"NPZ missing key '{key}': {npz_path}")
    return _basename_list(z[key])


def load_pi3_npz(npz_path: Path) -> dict[str, np.ndarray]:
    """Load Pi3 NPZ -> basename -> c2w (4x4) float64."""

    z = np.load(npz_path, allow_pickle=True)
    if "camera_poses" not in z:
        raise KeyError("NPZ missing key 'camera_poses'")
    if "image_paths" not in z:
        raise KeyError("NPZ missing key 'image_paths'")

    poses = z["camera_poses"]  # (N,4,4) c2w
    paths = z["image_paths"]

    out: dict[str, np.ndarray] = {}
    for T, p in zip(poses, paths, strict=True):
        out[Path(str(p)).name] = np.asarray(T, dtype=np.float64)
    return out


def load_centers_npz(npz_path: str | Path, *, key: str = "centers") -> dict[str, np.ndarray]:
    """Load camera centers from an NPZ file."""

    z = np.load(str(npz_path), allow_pickle=True)
    if "image_paths" not in z:
        raise KeyError(f"NPZ missing key 'image_paths': {npz_path}")
    names = _basename_list(z["image_paths"])

    if key not in z:
        raise KeyError(f"NPZ missing key '{key}': {npz_path}")

    arr = np.asarray(z[key], dtype=np.float64)

    # Direct centers
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = arr.reshape(-1, 3)
    if arr.ndim == 2 and arr.shape[1] == 3:
        if len(names) != arr.shape[0]:
            raise ValueError(f"image_paths length {len(names)} != centers N {arr.shape[0]}")
        return {n: arr[i].reshape(3) for i, n in enumerate(names)}

    # Extrinsics -> centers (assume w2c)
    if arr.ndim == 4 and arr.shape[-2:] == (3, 4):
        arr = arr.reshape(-1, 3, 4)
    if arr.ndim == 3 and arr.shape[-2:] == (3, 4):
        if len(names) != arr.shape[0]:
            raise ValueError(f"image_paths length {len(names)} != extrinsics N {arr.shape[0]}")
        R = arr[:, :3, :3]
        t = arr[:, :3, 3]
        C = -(np.transpose(R, (0, 2, 1)) @ t[..., None]).squeeze(-1)
        return {n: C[i].reshape(3) for i, n in enumerate(names)}

    if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
        if len(names) != arr.shape[0]:
            raise ValueError(f"image_paths length {len(names)} != extrinsics N {arr.shape[0]}")
        R = arr[:, :3, :3]
        t = arr[:, :3, 3]
        C = -(np.transpose(R, (0, 2, 1)) @ t[..., None]).squeeze(-1)
        return {n: C[i].reshape(3) for i, n in enumerate(names)}

    raise ValueError(f"Unsupported '{key}' shape {arr.shape} in {npz_path}")


def load_da3_npz_centers(npz_path: Path) -> dict[str, np.ndarray]:
    """Load DA3 NPZ camera centers (try multiple possible keys)."""

    for key in ("centers", "camera_centers", "C_world", "extrinsics_raw", "extrinsics"):
        try:
            return load_centers_npz(npz_path, key=key)
        except KeyError:
            continue

    raise KeyError(
        f"DA3 NPZ missing any supported center/extrinsics key in {npz_path}. "
        "Tried: centers, camera_centers, C_world, extrinsics_raw, extrinsics"
    )

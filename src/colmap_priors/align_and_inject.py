from __future__ import annotations

from pathlib import Path

import numpy as np

from . import sim3
from .colmap_db import write_position_priors, write_priors
from .colmap_model import ensure_txt_model, load_images_txt_centers
from .geometry import PoseSet, PoseW2C, c2w_to_w2c_and_center, rotmat_to_qvec
from .npz_io import load_pi3_npz
from .sim3 import apply_to_c2w


def align_c2w_to_reference(
    c2w_by_name: dict[str, np.ndarray],
    reference_model: Path,
    *,
    tmp_txt_dir: Path | None = None,
    colmap_exe: str = "colmap",
    min_pairs: int = 10,
) -> tuple[PoseSet, tuple[float, np.ndarray, np.ndarray]]:
    ref_txt_dir = ensure_txt_model(reference_model, out_dir=tmp_txt_dir, colmap_exe=colmap_exe)
    ref_centers = load_images_txt_centers(ref_txt_dir / "images.txt")

    common = sorted(set(c2w_by_name.keys()) & set(ref_centers.keys()))
    if len(common) < min_pairs:
        raise RuntimeError(f"Not enough overlap for alignment: {len(common)} < {min_pairs}")

    C_src = np.stack([c2w_by_name[n][:3, 3] for n in common], axis=0)
    C_ref = np.stack([ref_centers[n] for n in common], axis=0)
    s, R, t = sim3.umeyama(C_ref, C_src)  # maps src -> ref

    poses: dict[str, PoseW2C] = {}
    for name, T_c2w in c2w_by_name.items():
        T_aligned = apply_to_c2w(T_c2w, s, R, t)
        R_w2c, t_w2c, _ = c2w_to_w2c_and_center(T_aligned)
        q = rotmat_to_qvec(R_w2c)
        poses[name] = PoseW2C(qvec=q, tvec=t_w2c.astype(np.float64))

    return PoseSet(poses), (s, R, t)


def align_centers_to_reference(
    centers_by_name: dict[str, np.ndarray],
    reference_model: Path,
    *,
    tmp_txt_dir: Path | None = None,
    colmap_exe: str = "colmap",
    min_pairs: int = 10,
) -> tuple[dict[str, np.ndarray], tuple[float, np.ndarray, np.ndarray]]:
    ref_txt_dir = ensure_txt_model(reference_model, out_dir=tmp_txt_dir, colmap_exe=colmap_exe)
    ref_centers = load_images_txt_centers(ref_txt_dir / "images.txt")

    common = sorted(set(centers_by_name.keys()) & set(ref_centers.keys()))
    if len(common) < min_pairs:
        raise RuntimeError(f"Not enough overlap for alignment: {len(common)} < {min_pairs}")

    C_src = np.stack([centers_by_name[n] for n in common], axis=0)
    C_ref = np.stack([ref_centers[n] for n in common], axis=0)
    s, R, t = sim3.umeyama(C_ref, C_src)  # maps src -> ref

    aligned = {n: (s * (R @ np.asarray(C).reshape(3)) + t) for n, C in centers_by_name.items()}
    return aligned, (s, R, t)


def align_pi3_to_reference(
    pi3_npz: Path,
    reference_model: Path,
    *,
    tmp_txt_dir: Path | None = None,
    colmap_exe: str = "colmap",
    min_pairs: int = 10,
) -> tuple[PoseSet, tuple[float, np.ndarray, np.ndarray]]:
    c2w_by_name = load_pi3_npz(pi3_npz)
    return align_c2w_to_reference(
        c2w_by_name,
        reference_model,
        tmp_txt_dir=tmp_txt_dir,
        colmap_exe=colmap_exe,
        min_pairs=min_pairs,
    )


def inject_aligned_pi3(
    db_path: Path,
    pi3_npz: Path,
    reference_model: Path,
    *,
    strict: bool = True,
    tmp_txt_dir: Path | None = None,
    colmap_exe: str = "colmap",
    min_pairs: int = 10,
) -> tuple[float, np.ndarray, np.ndarray]:
    poseset, (s, R, t) = align_pi3_to_reference(
        pi3_npz,
        reference_model,
        tmp_txt_dir=tmp_txt_dir,
        colmap_exe=colmap_exe,
        min_pairs=min_pairs,
    )
    write_priors(db_path, poseset, strict=strict)
    return s, R, t


def inject_aligned_centers(
    db_path: Path,
    centers_by_name: dict[str, np.ndarray],
    reference_model: Path,
    *,
    strict: bool = True,
    tmp_txt_dir: Path | None = None,
    colmap_exe: str = "colmap",
    min_pairs: int = 10,
) -> tuple[float, np.ndarray, np.ndarray]:
    aligned, (s, R, t) = align_centers_to_reference(
        centers_by_name,
        reference_model,
        tmp_txt_dir=tmp_txt_dir,
        colmap_exe=colmap_exe,
        min_pairs=min_pairs,
    )
    write_position_priors(db_path, aligned, strict=strict)
    return s, R, t

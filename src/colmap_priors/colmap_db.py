from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Iterable

import numpy as np

from .poses import PoseSet
from .geometry import qvec_to_rotmat


PRIOR_Q_COLS = ["prior_qw", "prior_qx", "prior_qy", "prior_qz"]
PRIOR_T_COLS = ["prior_tx", "prior_ty", "prior_tz"]


def _has_column(cur: sqlite3.Cursor, table: str, column: str) -> bool:
    cur.execute(f"PRAGMA table_info({table});")
    return any(row[1] == column for row in cur.fetchall())


def ensure_prior_columns(db_path: Path) -> None:
    """Ensure images.prior_q* and images.prior_t* columns exist (works with stock COLMAP DB schema)."""
    db_path = Path(db_path)
    with _connect(db_path) as con:
        cur = con.cursor()
        for col in PRIOR_Q_COLS + PRIOR_T_COLS:
            if not _has_column(cur, "images", col):
                cur.execute(f"ALTER TABLE images ADD COLUMN {col} REAL;")
        con.commit()


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA foreign_keys=ON;")
    return con


def _table_info(cur: sqlite3.Cursor, table: str) -> list[str]:
    cur.execute(f"PRAGMA table_info({table});")
    return [r[1] for r in cur.fetchall()]


def _has_table(cur: sqlite3.Cursor, table: str) -> bool:
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;", (table,))
    return cur.fetchone() is not None


def _pack_f64(vec: Iterable[float]) -> bytes:
    a = np.asarray(list(vec), dtype=np.float64).reshape(-1)
    return a.tobytes(order="C")


def _camera_center_from_w2c(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R = qvec_to_rotmat(qvec)
    C = -(R.T @ np.asarray(tvec, dtype=np.float64).reshape(3))
    return C


def write_priors(db_path: Path, poseset: PoseSet, *, strict: bool = True) -> None:
    """Write pose priors.

    IMPORTANT: COLMAP's pose_prior_mapper expects *position priors* (camera centers in world
    coordinates). We therefore store the camera center (C) in prior_tx/ty/tz and, when
    available, also insert it into the 'pose_priors' table (COLMAP >= 3.14 dev).
    """
    ensure_prior_columns(db_path)
    db_path = Path(db_path)

    with _connect(db_path) as con:
        cur = con.cursor()

        # Map image name -> id once.
        cur.execute("SELECT image_id, name FROM images;")
        name_to_id = {Path(n).name: int(i) for i, n in cur.fetchall()}

        # Introspect schema (COLMAP variants differ)
        image_cols = set(_table_info(cur, "images"))

        has_pos = {"prior_tx", "prior_ty", "prior_tz"}.issubset(image_cols)
        has_quat = {"prior_qw", "prior_qx", "prior_qy", "prior_qz"}.issubset(image_cols)

        missing = []

        for name, p in poseset.poses.items():
            image_id = name_to_id.get(name)
            if image_id is None:
                missing.append(name)
                continue

            C = _camera_center_from_w2c(p.qvec, p.tvec)

            if has_pos and has_quat:
                cur.execute(
                    """
                    UPDATE images SET
                      prior_qw=?, prior_qx=?, prior_qy=?, prior_qz=?,
                      prior_tx=?, prior_ty=?, prior_tz=?
                    WHERE image_id=?;
                    """,
                    (
                        float(p.qvec[0]), float(p.qvec[1]), float(p.qvec[2]), float(p.qvec[3]),
                        float(C[0]), float(C[1]), float(C[2]),
                        image_id,
                    ),
                )
            elif has_pos:
                # Only write position priors (robust across COLMAP schemas)
                cur.execute(
                    """
                    UPDATE images SET
                      prior_tx=?, prior_ty=?, prior_tz=?
                    WHERE image_id=?;
                    """,
                    (float(C[0]), float(C[1]), float(C[2]), image_id),
                )
            else:
                pass

        if strict and missing:
            raise RuntimeError(
                f"{len(missing)} priors did not match any COLMAP image name. Example: {missing[0]}"
            )

        # Also populate pose_priors table when present (COLMAP >= 3.14 dev).
        if _has_table(cur, "pose_priors"):
            cols = _table_info(cur, "pose_priors")
            pos_col = "position" if "position" in cols else None
            cov_col = None
            for c in ("position_covariance", "covariance"):
                if c in cols:
                    cov_col = c
                    break
            coord_col = "coordinate_system" if "coordinate_system" in cols else None

            if pos_col is not None:
                cov_blob = _pack_f64(np.eye(3, dtype=np.float64).reshape(-1))
                coord_val = 1  # CARTESIAN

                for name, p in poseset.poses.items():
                    image_id = name_to_id.get(name)
                    if image_id is None:
                        continue
                    C = _camera_center_from_w2c(p.qvec, p.tvec)
                    pos_blob = _pack_f64(C)

                    if cov_col and coord_col:
                        cur.execute(
                            f"INSERT OR REPLACE INTO pose_priors (image_id, {pos_col}, {cov_col}, {coord_col}) VALUES (?, ?, ?, ?);",
                            (image_id, pos_blob, cov_blob, coord_val),
                        )
                    elif cov_col:
                        cur.execute(
                            f"INSERT OR REPLACE INTO pose_priors (image_id, {pos_col}, {cov_col}) VALUES (?, ?, ?);",
                            (image_id, pos_blob, cov_blob),
                        )
                    elif coord_col:
                        cur.execute(
                            f"INSERT OR REPLACE INTO pose_priors (image_id, {pos_col}, {coord_col}) VALUES (?, ?, ?);",
                            (image_id, pos_blob, coord_val),
                        )
                    else:
                        cur.execute(
                            f"INSERT OR REPLACE INTO pose_priors (image_id, {pos_col}) VALUES (?, ?);",
                            (image_id, pos_blob),
                        )

        con.commit()


def write_position_priors(
    db_path: Path,
    centers_by_name: dict[str, np.ndarray],
    *,
    strict: bool = True,
) -> None:
    """Write position priors only (camera centers in world coordinates)."""
    ensure_prior_columns(db_path)
    db_path = Path(db_path)

    with _connect(db_path) as con:
        cur = con.cursor()
        cur.execute("SELECT image_id, name FROM images;")
        name_to_id = {Path(n).name: int(i) for i, n in cur.fetchall()}

        missing = []
        for name, C in centers_by_name.items():
            image_id = name_to_id.get(name)
            if image_id is None:
                missing.append(name)
                continue

            C = np.asarray(C, dtype=np.float64).reshape(3)
            cur.execute(
                """UPDATE images SET prior_tx=?, prior_ty=?, prior_tz=? WHERE image_id=?;""",
                (float(C[0]), float(C[1]), float(C[2]), image_id),
            )

        if strict and missing:
            raise RuntimeError(
                f"{len(missing)} priors did not match any COLMAP image name. Example: {missing[0]}"
            )

        if _has_table(cur, "pose_priors"):
            cols = _table_info(cur, "pose_priors")
            pos_col = "position" if "position" in cols else None
            cov_col = None
            for c in ("position_covariance", "covariance"):
                if c in cols:
                    cov_col = c
                    break
            coord_col = "coordinate_system" if "coordinate_system" in cols else None

            if pos_col is not None:
                cov_blob = _pack_f64(np.eye(3, dtype=np.float64).reshape(-1))
                coord_val = 1

                for name, C in centers_by_name.items():
                    image_id = name_to_id.get(name)
                    if image_id is None:
                        continue
                    C = np.asarray(C, dtype=np.float64).reshape(3)
                    pos_blob = _pack_f64(C)

                    if cov_col and coord_col:
                        cur.execute(
                            f"INSERT OR REPLACE INTO pose_priors (image_id, {pos_col}, {cov_col}, {coord_col}) VALUES (?, ?, ?, ?);",
                            (image_id, pos_blob, cov_blob, coord_val),
                        )
                    elif cov_col:
                        cur.execute(
                            f"INSERT OR REPLACE INTO pose_priors (image_id, {pos_col}, {cov_col}) VALUES (?, ?, ?);",
                            (image_id, pos_blob, cov_blob),
                        )
                    elif coord_col:
                        cur.execute(
                            f"INSERT OR REPLACE INTO pose_priors (image_id, {pos_col}, {coord_col}) VALUES (?, ?, ?);",
                            (image_id, pos_blob, coord_val),
                        )
                    else:
                        cur.execute(
                            f"INSERT OR REPLACE INTO pose_priors (image_id, {pos_col}) VALUES (?, ?);",
                            (image_id, pos_blob),
                        )

        con.commit()
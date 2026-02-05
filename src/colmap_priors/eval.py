from __future__ import annotations
import os

from pathlib import Path
import re

import numpy as np

from .colmap_model import ensure_txt_model, load_images_txt_centers
from .align_and_inject import align_centers_to_reference
from .npz_io import (
    load_npz_order,
    load_pi3_npz,
    load_da3_npz_centers,
)


def _nat_key(s: str):
    parts = re.split(r"(\d+)", s)
    out = []
    for p in parts:
        out.append(int(p) if p.isdigit() else p.lower())
    return out


def _sanitize_label(label: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", label.strip())
    return s.strip("_").lower()


def _best_plane(xyz: np.ndarray) -> str:
    if xyz.shape[0] < 2:
        return "xy"
    v = np.var(xyz, axis=0)
    score_xy = float(v[0] * v[1])
    score_xz = float(v[0] * v[2])
    score_yz = float(v[1] * v[2])
    if score_xz >= score_xy and score_xz >= score_yz:
        return "xz"
    if score_yz >= score_xy and score_yz >= score_xz:
        return "yz"
    return "xy"


def _project(xyz: np.ndarray, plane: str) -> tuple[np.ndarray, np.ndarray]:
    if plane == "xy":
        return xyz[:, 0], xyz[:, 1]
    if plane == "xz":
        return xyz[:, 0], xyz[:, 2]
    if plane == "yz":
        return xyz[:, 1], xyz[:, 2]
    raise ValueError(f"Unknown plane: {plane}")


def _rmse(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def colmap_frame_errors(model_txt_dir: Path) -> dict[str, float]:
    """Approx per-image reprojection error from COLMAP TXT model.

    Uses points3D.txt 'error' field, assigns it to all observations in each track,
    then averages per image. Returns basename -> mean reprojection error.
    """
    model_txt_dir = Path(model_txt_dir)
    images_file = model_txt_dir / "images.txt"
    points_file = model_txt_dir / "points3D.txt"
    if not images_file.exists() or not points_file.exists():
        return {}

    id_to_name: dict[int, str] = {}
    with images_file.open("r", encoding="utf-8") as f:
        it = iter(f)
        for raw in it:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                image_id = int(parts[0])
            except ValueError:
                continue
            id_to_name[image_id] = Path(parts[9]).name
            next(it, None)  # skip 2D points line

    err_sum: dict[int, float] = {}
    obs_count: dict[int, int] = {}
    with points_file.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                err = float(parts[7])
            except ValueError:
                continue

            track = parts[8:]
            for i in range(0, len(track), 2):
                try:
                    img_id = int(track[i])
                except ValueError:
                    continue
                err_sum[img_id] = err_sum.get(img_id, 0.0) + err
                obs_count[img_id] = obs_count.get(img_id, 0) + 1

    out: dict[str, float] = {}
    for img_id, total in err_sum.items():
        c = obs_count.get(img_id, 0)
        if c <= 0:
            continue
        name = id_to_name.get(img_id)
        if name is None:
            continue
        out[name] = total / float(c)
    return out


def _errors_for_names(model_txt_dir: Path | None, names: list[str]) -> tuple[np.ndarray | None, float | None]:
    if model_txt_dir is None:
        return None, None
    err_map = colmap_frame_errors(model_txt_dir)
    if not err_map:
        return None, None
    vals = np.asarray([err_map.get(n, np.nan) for n in names], dtype=np.float64)
    m = np.isfinite(vals)
    if not m.any():
        return None, None
    return vals, float(np.mean(vals[m]))


def load_reference_centers(
    reference_model: Path,
    *,
    tmp_txt_dir: Path | None = None,
    colmap_exe: str = "colmap",
) -> tuple[Path, list[str], dict[str, np.ndarray]]:
    ref_txt = ensure_txt_model(reference_model, out_dir=tmp_txt_dir, colmap_exe=colmap_exe)
    order, centers = load_images_txt_centers(ref_txt / "images.txt", return_order=True)
    order = sorted(centers.keys(), key=_nat_key)
    return ref_txt, order, centers


def _solid_color_for_label(label: str) -> str:
    l = label.lower()
    if l == "pi3_pre":
        return "red"
    if l == "da3_post":
        return "red"
    if l == "pi3_post":
        return "tab:blue"
    if l == "da3_pre":
        return "tab:orange"
    return "tab:green"


def _cmap_for_label(label: str) -> str:
    return "viridis"


def _plot_one(
    *,
    label: str,
    names: list[str],
    ref_xyz: np.ndarray,
    other_xyz: np.ndarray,
    rmse_center: float,
    per_image_err: np.ndarray | None,
    mean_err: float | None,
    out_path: Path,
    aspect: str = "auto",
    err_vmin: float | None = None,
    err_vmax: float | None = None,
    std_str: str | None = None,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    plane = _best_plane(ref_xyz)
    x_ref, y_ref = _project(ref_xyz, plane)
    x_o, y_o = _project(other_xyz, plane)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x_ref, y_ref, "--", linewidth=1.8, color="0.55", label="baseline")

    colored = False
    if per_image_err is not None and other_xyz.shape[0] >= 2:
        pts = np.stack([x_o, y_o], axis=1)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        seg_err = 0.5 * (per_image_err[:-1] + per_image_err[1:])

        m = np.isfinite(seg_err)
        if m.any():
            # Use provided global normalization if available; else fallback to local
            if err_vmin is None or err_vmax is None:
                vmin = float(np.nanmin(seg_err[m]))
                vmax = float(np.nanmax(seg_err[m]))
            else:
                vmin = float(err_vmin)
                vmax = float(err_vmax)

            if vmin == vmax:
                vmax = vmin + 1e-6

            cmap = plt.get_cmap(_cmap_for_label(label))
            norm = plt.Normalize(vmin=vmin, vmax=vmax)

            lc = LineCollection(segs, cmap=cmap, norm=norm)
            lc.set_array(seg_err)
            lc.set_linewidth(2.0)
            ax.add_collection(lc)

            # legend handle
            ax.plot([], [], "-", linewidth=2.0, color="0.2", label=label)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.85)
            cbar.set_label("Approx. per-image reprojection error")
            colored = True

    if not colored:
        ax.plot(x_o, y_o, "-", linewidth=2.0, color=_solid_color_for_label(label), label=label)

    title = f"baseline vs {label} | center-RMSE={rmse_center:.4g}"
    if mean_err is not None:
        title += f" | mean reproj={mean_err:.4g}"
    if std_str is not None:
        title += f" | std={std_str}"
    ax.set_title(f"{title}\n(aligned on {plane} plane; N={len(names)})")

    if aspect == "equal":
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("auto")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(loc="best")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_grid_plot(paths: list[Path], out_path: Path) -> Path:
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    k = len(paths)
    if k == 0:
        return out_path

    # grid: choose near-square
    cols = int(math.ceil(math.sqrt(k)))
    rows = int(math.ceil(k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    # Fill grid
    for i in range(rows * cols):
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        ax.axis("off")
        if i >= k:
            continue
        img = mpimg.imread(str(paths[i]))
        ax.imshow(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_trajectories(
    reference_model: Path,
    items: list[tuple[str, str, Path]],
    *,
    out_path: Path,
    min_pairs: int = 10,
    tmp_txt_dir: Path | None = None,
    colmap_exe: str = "colmap",
    with_errors: bool = True,
    aspect: str = "auto",
    ref_common_only: bool = True,
) -> Path:
    """One PNG per comparison item + an additional raster grid plot.

    - Alignment is delegated to align_and_inject.py so plotting uses the same logic as injection.
    - Reprojection-error colorbar scale is globally normalized across all kind=="model" plots.
    """
    out_path = Path(out_path)
    out_dir = out_path.parent
    stem = out_path.stem

    prior_std = os.environ.get("PRIOR_POSITION_STD")
    std_str = f"{float(prior_std):.4g}" if prior_std is not None else None

    # Load reference once
    _, ref_order, ref_centers = load_reference_centers(
        reference_model, tmp_txt_dir=tmp_txt_dir, colmap_exe=colmap_exe
    )

    by_label = {lbl: (lbl, kind, Path(p)) for (lbl, kind, p) in items}
    preferred = ["pi3_pre", "pi3_post", "da3_pre", "da3_post"]
    ordered_items = [by_label[lbl] for lbl in preferred if lbl in by_label]

    # ---------- Pre-pass: compute global error normalization over ALL model plots ----------
    global_err_vals: list[np.ndarray] = []
    if with_errors:
        for label, kind, path in ordered_items:
            if kind != "model":
                continue
            if not Path(path).exists():
                continue
            txt_dir = ensure_txt_model(path, out_dir=tmp_txt_dir, colmap_exe=colmap_exe)
            model_order, model_centers = load_images_txt_centers(txt_dir / "images.txt", return_order=True)
            aligned_centers, _ = align_centers_to_reference(
                model_centers,
                reference_model,
                tmp_txt_dir=tmp_txt_dir,
                colmap_exe=colmap_exe,
                min_pairs=min_pairs,
            )

            # common frames in model order
            common_names = [n for n in model_order if (n in ref_centers and n in aligned_centers)]
            if len(common_names) < min_pairs:
                continue

            if ref_common_only:
                ref_names = [n for n in ref_order if n in set(common_names)]
                if ref_names:
                    common_names = ref_names

            per_img, _ = _errors_for_names(txt_dir, common_names)
            if per_img is None:
                continue
            m = np.isfinite(per_img)
            if m.any():
                global_err_vals.append(per_img[m])

    if global_err_vals:
        allv = np.concatenate(global_err_vals, axis=0)
        err_vmin = float(np.nanmin(allv))
        err_vmax = float(np.nanmax(allv))
        if err_vmin == err_vmax:
            err_vmax = err_vmin + 1e-6
    else:
        err_vmin = None
        err_vmax = None

    # ---------- Main pass: create individual plots ----------
    created: list[Path] = []

    for label, kind, path in ordered_items:
        if not Path(path).exists():
            continue

        txt_dir: Path | None = None
        aligned_centers: dict[str, np.ndarray]

        if kind == "pi3":
            c2w = load_pi3_npz(path)
            centers_raw = {n: T[:3, 3].astype(np.float64) for n, T in c2w.items()}

            aligned_centers, _ = align_centers_to_reference(
                centers_raw,
                reference_model,
                tmp_txt_dir=tmp_txt_dir,
                colmap_exe=colmap_exe,
                min_pairs=min_pairs,
            )
            other_order = load_npz_order(path)

        elif kind == "da3":
            centers_raw = load_da3_npz_centers(path)
            aligned_centers, _ = align_centers_to_reference(
                centers_raw,
                reference_model,
                tmp_txt_dir=tmp_txt_dir,
                colmap_exe=colmap_exe,
                min_pairs=min_pairs,
            )
            other_order = load_npz_order(path)

        elif kind == "model":
            txt_dir = ensure_txt_model(path, out_dir=tmp_txt_dir, colmap_exe=colmap_exe)
            model_order, model_centers = load_images_txt_centers(txt_dir / "images.txt", return_order=True)
            aligned_centers, _ = align_centers_to_reference(
                model_centers,
                reference_model,
                tmp_txt_dir=tmp_txt_dir,
                colmap_exe=colmap_exe,
                min_pairs=min_pairs,
            )
            other_order = model_order

        else:
            raise ValueError(f"Unknown kind: {kind}")

        # Common frames, preserve other ordering
        common_names = [n for n in other_order if (n in ref_centers and n in aligned_centers)]
        if len(common_names) < min_pairs:
            raise RuntimeError(f"{label}: too few overlapping images: {len(common_names)} < {min_pairs}")

        if ref_common_only:
            ref_names = [n for n in ref_order if n in set(common_names)]
            if ref_names:
                common_names = ref_names

        ref_xyz = np.stack([ref_centers[n] for n in common_names], axis=0)
        other_xyz = np.stack([aligned_centers[n] for n in common_names], axis=0)
        rmse_center = _rmse(ref_xyz, other_xyz)

        per_img = None
        mean_err = None
        if with_errors and kind == "model":
            per_img, mean_err = _errors_for_names(txt_dir, common_names)

        out_file = out_dir / f"{stem}_{_sanitize_label(label)}.png"
        _plot_one(
            label=label,
            names=common_names,
            ref_xyz=ref_xyz,
            other_xyz=other_xyz,
            rmse_center=rmse_center,
            per_image_err=per_img,
            mean_err=mean_err,
            out_path=out_file,
            aspect=aspect,
            err_vmin=err_vmin,
            err_vmax=err_vmax,
            std_str=std_str,
        )
        created.append(out_file)

    # ---------- Additional grid plot ----------
    if created:
        grid_file = out_dir / f"{stem}_grid.png"
        _write_grid_plot(created, grid_file)

    return created[0] if created else out_path
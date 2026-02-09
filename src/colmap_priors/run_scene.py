from __future__ import annotations

import argparse
import subprocess
from collections.abc import Iterable
from pathlib import Path

from .align_and_inject import inject_aligned_centers, inject_aligned_pi3
from .colmap_cli import (
    database_creator,
    exhaustive_matcher,
    feature_extractor,
    mapper,
    model_to_txt,
    pose_prior_mapper,
)
from .config import load_config
from .eval import plot_trajectories
from .npz_io import load_da3_npz_centers

# ---------------------------------------------------------------------------
# Image helpers (inlined from former select_images.py)
# ---------------------------------------------------------------------------

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _count_images(image_dir: Path) -> int:
    return sum(1 for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _write_image_list(image_dir: Path, *, max_images: int, out_path: Path) -> list[str]:
    files = sorted(p.name for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if max_images <= 0 or max_images >= len(files):
        sel = files
    elif max_images == 1:
        sel = [files[0]]
    else:
        step = (len(files) - 1) / float(max_images - 1)
        idx = [round(i * step) for i in range(max_images)]
        sel = [files[min(int(j), len(files) - 1)] for j in idx]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sel) + "\n", encoding="utf-8")
    return sel


def _make_symlink_dir(src_dir: Path, names: Iterable[str], dst_dir: Path) -> Path:
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in dst_dir.iterdir():
        if p.is_file() or p.is_symlink():
            p.unlink()
    for name in names:
        src = src_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Listed image not found: {src}")
        (dst_dir / name).symlink_to(src)
    return dst_dir


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _run_export(python_bin: str, model: str, args: list[str]) -> None:
    script = _SCRIPTS_DIR / "export.py"
    subprocess.run([python_bin, str(script), model, *args], check=True, stdout=None, stderr=None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run scene pipeline (baseline/Pi3/DA3/plot) with COLMAP CLI.")
    ap.add_argument("scene")
    ap.add_argument("--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--colmap", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.colmap:
        cfg.colmap_exe = args.colmap

    scene = args.scene
    src_scene_dir = cfg.data_root / scene
    src_image_dir = src_scene_dir / "images"
    if not src_image_dir.is_dir():
        raise FileNotFoundError(f"Source images not found: {src_image_dir}")

    scene_root = cfg.exp_root / scene
    scene_root.mkdir(parents=True, exist_ok=True)

    list_path = scene_root / "image_list.txt"
    base_db = scene_root / "base.db"

    baseline_sparse = scene_root / "colmap_baseline" / "sparse"
    baseline_txt = baseline_sparse / "0_txt"

    pi3_dir = scene_root / "pi3"
    pi3_npz = pi3_dir / "pi3_predictions.npz"
    pi3_db = pi3_dir / "database.db"
    pi3_sparse = pi3_dir / "sparse"
    pi3_txt = pi3_sparse / "0_txt"

    da3_dir = scene_root / "da3"
    da3_npz = da3_dir / "da3_predictions.npz"
    da3_db = da3_dir / "database.db"
    da3_sparse = da3_dir / "sparse"
    da3_txt = da3_sparse / "0_txt"

    ref_model_dir = cfg.ref_model_dir or (src_scene_dir / "sparse" / "0")

    print(f"=== Scene: {scene} ===")
    print(f"images: {src_image_dir}")
    print(f"out   : {scene_root}")

    # 1) image list
    total = _count_images(src_image_dir)
    names = _write_image_list(src_image_dir, max_images=cfg.max_images, out_path=list_path)
    print(f"[1/6] image list: {len(names)}/{total}")

    infer_dir = src_image_dir
    if 0 < cfg.max_images < total:
        infer_dir = _make_symlink_dir(src_image_dir, names, scene_root / "_inference_images")

    # 2) base db
    if base_db.exists():
        print(f"[2/6] base DB exists: {base_db}")
        print("[2/6] Skipping database creation, feature extraction, and matching.")
    else:
        print(f"[2/6] creating base DB: {base_db}")
        database_creator(base_db, colmap_exe=cfg.colmap_exe)
        image_list_for_colmap = list_path if (0 < cfg.max_images < total) else None
        feature_extractor(base_db, src_image_dir, image_list_path=image_list_for_colmap, colmap_exe=cfg.colmap_exe)
        exhaustive_matcher(base_db, colmap_exe=cfg.colmap_exe)

    # 3) baseline
    if cfg.run_baseline:
        print("[3/6] baseline mapper")
        baseline_sparse.mkdir(parents=True, exist_ok=True)
        mapper(base_db, src_image_dir, baseline_sparse, colmap_exe=cfg.colmap_exe)
        model_to_txt(baseline_sparse / "0", baseline_txt, colmap_exe=cfg.colmap_exe)
        if cfg.ref_model_dir is None:
            ref_model_dir = baseline_sparse / "0"
    else:
        print("[3/6] baseline mapper: skipped")

    if not ref_model_dir.exists():
        raise FileNotFoundError(
            f"Reference model not found: {ref_model_dir}. Set ref_model_dir or enable run_baseline."
        )

    # 4) pi3
    if cfg.run_pi3:
        print("[4/6] Pi3: export -> inject -> pose_prior_mapper")
        pi3_dir.mkdir(parents=True, exist_ok=True)
        if pi3_db.exists():
            pi3_db.unlink()
        pi3_db.write_bytes(base_db.read_bytes())

        _run_export(
            cfg.pi3_python,
            "pi3",
            ["--image_dir", str(infer_dir), "--output", str(pi3_npz), "--interval", str(cfg.pi3_interval)],
        )

        inject_aligned_pi3(
            pi3_db,
            pi3_npz,
            ref_model_dir,
            strict=True,
            tmp_txt_dir=scene_root / "_tmp_ref_txt",
            colmap_exe=cfg.colmap_exe,
        )

        pi3_sparse.mkdir(parents=True, exist_ok=True)
        pose_prior_mapper(
            pi3_db, src_image_dir, pi3_sparse, prior_std=cfg.prior_position_std, colmap_exe=cfg.colmap_exe
        )
        model_to_txt(pi3_sparse / "0", pi3_txt, colmap_exe=cfg.colmap_exe)
    else:
        print("[4/6] Pi3: skipped")

    # 5) da3
    if cfg.run_da3:
        print("[5/6] DA3: export -> inject centers -> pose_prior_mapper")
        da3_dir.mkdir(parents=True, exist_ok=True)
        if da3_db.exists():
            da3_db.unlink()
        da3_db.write_bytes(base_db.read_bytes())

        da3_args = [
            "--image_dir",
            str(infer_dir),
            "--output",
            str(da3_npz),
            "--window",
            str(cfg.da3_window),
            "--overlap",
            str(cfg.da3_overlap),
        ]
        if cfg.da3_autocast:
            da3_args.append("--autocast")
        _run_export(cfg.da3_python, "da3", da3_args)

        centers = load_da3_npz_centers(da3_npz)

        inject_aligned_centers(
            da3_db,
            centers,
            ref_model_dir,
            strict=True,
            tmp_txt_dir=scene_root / "_tmp_ref_txt",
            colmap_exe=cfg.colmap_exe,
        )

        da3_sparse.mkdir(parents=True, exist_ok=True)
        pose_prior_mapper(
            da3_db, src_image_dir, da3_sparse, prior_std=cfg.prior_position_std, colmap_exe=cfg.colmap_exe
        )
        model_to_txt(da3_sparse / "0", da3_txt, colmap_exe=cfg.colmap_exe)
    else:
        print("[5/6] DA3: skipped")

    # 6) plot
    if cfg.run_plot:
        print("[6/6] plot trajectories")
        out_plot = scene_root / "plots" / "trajectories.png"
        items: list[tuple[str, str, Path]] = []
        if pi3_npz.exists():
            items.append(("pi3_pre", "pi3", pi3_npz))
        if (pi3_sparse / "0").exists():
            items.append(("pi3_post", "model", pi3_sparse / "0"))
        elif pi3_txt.exists():
            items.append(("pi3_post", "model", pi3_txt))
        if da3_npz.exists():
            items.append(("da3_pre", "da3", da3_npz))
        if (da3_sparse / "0").exists():
            items.append(("da3_post", "model", da3_sparse / "0"))
        elif da3_txt.exists():
            items.append(("da3_post", "model", da3_txt))

        plot_trajectories(
            ref_model_dir,
            items,
            out_path=out_plot,
            tmp_txt_dir=scene_root / "_tmp_plot_txt",
            colmap_exe=cfg.colmap_exe,
            with_errors=True,
            aspect="auto",
            ref_common_only=True,
            prior_std=cfg.prior_position_std,
        )
        print(f"plot: {out_plot}")
    else:
        print("[6/6] plot: skipped")

    return 0

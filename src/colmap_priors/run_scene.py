from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Iterable

from .config import load_env_file
from .select_images import IMAGE_EXTS, write_image_list
from .colmap_cli import (
    database_creator,
    feature_extractor,
    exhaustive_matcher,
    mapper,
    model_to_txt,
    pose_prior_mapper,
)
from .align_and_inject import inject_aligned_pi3, inject_aligned_centers
from .npz_io import load_centers_npz
from .eval import plot_trajectories


def _count_images(image_dir: Path) -> int:
    return sum(1 for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


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


def _run_export(python_bin: str, script: Path, args: list[str]) -> None:
    subprocess.run([python_bin, str(script), *args], check=True, stdout=None, stderr=None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run scene pipeline (baseline/Pi3/DA3/plot) with COLMAP CLI.")
    ap.add_argument("scene")
    ap.add_argument("--env", type=Path, default=Path("scripts/colmap_priors.env"))
    ap.add_argument("--colmap", default=None)
    args = ap.parse_args(argv)

    load_env_file(args.env)

    scene = args.scene
    data_root = Path(os.environ["DATA_ROOT"])
    exp_root = Path(os.environ["EXP_ROOT"])
    max_images = int(os.environ.get("MAX_IMAGES", "0"))
    prior_std = float(os.environ["PRIOR_POSITION_STD"])

    run_baseline = os.environ.get("RUN_BASELINE_MAP", "0") == "1"
    run_pi3 = os.environ.get("RUN_PI3", "1") == "1"
    run_da3 = os.environ.get("RUN_DA3", "0") == "1"
    run_plot = os.environ.get("RUN_PLOT", "1") == "1"

    colmap_exe = args.colmap or os.environ.get("COLMAP_EXE", "colmap")

    pi3_repo = os.environ.get("PI3_REPO", "")
    da3_repo = os.environ.get("DA3_REPO", "")

    pi3_python = os.environ.get("PI3_PYTHON", "python")
    pi3_interval = os.environ.get("PI3_INTERVAL", "1")

    da3_python = os.environ.get("DA3_PYTHON", "python")
    da3_window = os.environ.get("DA3_WINDOW", "10")
    da3_overlap = os.environ.get("DA3_OVERLAP", "6")
    da3_autocast = os.environ.get("DA3_AUTocast", "1") == "1"

    src_scene_dir = data_root / scene
    src_image_dir = src_scene_dir / "images"
    if not src_image_dir.is_dir():
        raise FileNotFoundError(f"Source images not found: {src_image_dir}")

    scene_root = exp_root / scene
    scene_root.mkdir(parents=True, exist_ok=True)

    list_path = scene_root / "image_list.txt"
    base_db = scene_root / "base.db"

    baseline_dir = scene_root / "colmap_baseline"
    baseline_sparse = baseline_dir / "sparse"
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

    ref_model_dir = Path(os.environ.get("REF_MODEL_DIR", str(src_scene_dir / "sparse" / "0")))

    print(f"=== Scene: {scene} ===")
    print(f"images: {src_image_dir}")
    print(f"out   : {scene_root}")

    # 1) image list
    total = _count_images(src_image_dir)
    names = write_image_list(src_image_dir, max_images=max_images, out_path=list_path)
    print(f"[1/6] image list: {len(names)}/{total}")

    # inference dir (symlink subset only if strict subset)
    infer_dir = src_image_dir
    if 0 < max_images < total:
        infer_dir = _make_symlink_dir(src_image_dir, names, scene_root / "_inference_images")

    # 2) base db
    if base_db.exists():
        print(f"[2/6] base DB exists: {base_db}")
        print("[2/6] Skipping database creation, feature extraction, and matching.")
    else:
        print(f"[2/6] creating base DB: {base_db}")
        database_creator(base_db, colmap_exe=colmap_exe)
        image_list_for_colmap = list_path if (0 < max_images < total) else None
        feature_extractor(base_db, src_image_dir, image_list_path=image_list_for_colmap, colmap_exe=colmap_exe)
        exhaustive_matcher(base_db, colmap_exe=colmap_exe)

    # 3) baseline
    if run_baseline:
        print("[3/6] baseline mapper")
        baseline_sparse.mkdir(parents=True, exist_ok=True)
        mapper(base_db, src_image_dir, baseline_sparse, colmap_exe=colmap_exe)
        model_to_txt(baseline_sparse / "0", baseline_txt, colmap_exe=colmap_exe)
        if "REF_MODEL_DIR" not in os.environ:
            ref_model_dir = baseline_sparse / "0"
    else:
        print("[3/6] baseline mapper: skipped")

    if not ref_model_dir.exists():
        raise FileNotFoundError(
            f"Reference model not found: {ref_model_dir}. Set REF_MODEL_DIR or enable RUN_BASELINE_MAP=1."
        )

    # 4) pi3
    if run_pi3:
        if not pi3_repo:
            raise RuntimeError("RUN_PI3=1 but PI3_REPO is not set.")
        print("[4/6] Pi3: export -> inject -> pose_prior_mapper")
        pi3_dir.mkdir(parents=True, exist_ok=True)
        if pi3_db.exists():
            pi3_db.unlink()
        pi3_db.write_bytes(base_db.read_bytes())

        _run_export(pi3_python, Path(__file__).resolve().parents[2] / "scripts" / "export_pi3_poses.py",
            ["--image_dir", str(infer_dir), "--output", str(pi3_npz), "--interval", str(pi3_interval)],
        )

        inject_aligned_pi3(
            pi3_db,
            pi3_npz,
            ref_model_dir,
            strict=True,
            tmp_txt_dir=scene_root / "_tmp_ref_txt",
            colmap_exe=colmap_exe,
        )

        pi3_sparse.mkdir(parents=True, exist_ok=True)
        pose_prior_mapper(pi3_db, src_image_dir, pi3_sparse, prior_std=prior_std, colmap_exe=colmap_exe)
        model_to_txt(pi3_sparse / "0", pi3_txt, colmap_exe=colmap_exe)
    else:
        print("[4/6] Pi3: skipped")

    # 5) da3
    if run_da3:
        if not da3_repo:
            raise RuntimeError("RUN_DA3=1 but DA3_REPO is not set.")
        print("[5/6] DA3: export -> inject centers -> pose_prior_mapper")
        da3_dir.mkdir(parents=True, exist_ok=True)
        if da3_db.exists():
            da3_db.unlink()
        da3_db.write_bytes(base_db.read_bytes())

        da3_args = ["--image_dir", str(infer_dir), "--output", str(da3_npz), "--window", str(da3_window), "--overlap", str(da3_overlap)]
        if da3_autocast:
            da3_args.append("--autocast")
        _run_export(da3_python, Path(__file__).resolve().parents[2] / "scripts" / "export_da3_poses.py", da3_args)

        centers = None
        for key in ("centers", "camera_centers", "C_world", "extrinsics", "extrinsics_raw"):
            try:
                centers = load_centers_npz(da3_npz, key=key)
                break
            except KeyError:
                continue
        if centers is None:
            raise KeyError(f"No centers key found in {da3_npz}. Tried: centers, camera_centers, C_world, extrinsics")

        inject_aligned_centers(
            da3_db,
            centers,
            ref_model_dir,
            strict=True,
            tmp_txt_dir=scene_root / "_tmp_ref_txt",
            colmap_exe=colmap_exe,
        )

        da3_sparse.mkdir(parents=True, exist_ok=True)
        pose_prior_mapper(da3_db, src_image_dir, da3_sparse, prior_std=prior_std, colmap_exe=colmap_exe)
        model_to_txt(da3_sparse / "0", da3_txt, colmap_exe=colmap_exe)
    else:
        print("[5/6] DA3: skipped")

    # 6) plot
    if run_plot:
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
            colmap_exe=colmap_exe,
            with_errors=True,
            aspect="auto",
            ref_common_only=True,
        )
        print(f"plot: {out_plot}")
    else:
        print("[6/6] plot: skipped")

    return 0
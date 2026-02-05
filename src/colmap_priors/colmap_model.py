from __future__ import annotations

from pathlib import Path
import subprocess
import numpy as np

from .geometry import qvec_to_rotmat


def ensure_txt_model(model_dir: Path, *, out_dir: Path | None = None, colmap_exe: str = "colmap") -> Path:
    model_dir = Path(model_dir)
    if (model_dir / "images.txt").exists():
        return model_dir

    # BIN model
    if not (model_dir / "images.bin").exists():
        raise FileNotFoundError(f"Not a COLMAP model dir (no images.txt/bin): {model_dir}")

    out_dir = Path(out_dir) if out_dir is not None else (model_dir.parent / (model_dir.name + "_txt"))
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        colmap_exe, "model_converter",
        "--input_path", str(model_dir),
        "--output_path", str(out_dir),
        "--output_type", "TXT",
    ], check=True)
    return out_dir


def load_images_txt_centers(
    images_txt: Path,
    *,
    return_order: bool = False,
) -> dict[str, np.ndarray] | tuple[list[str], dict[str, np.ndarray]]:
    """Load camera centers from COLMAP images.txt file."""
    images_txt = Path(images_txt)
    centers: dict[str, np.ndarray] = {}
    order: list[str] = []

    with images_txt.open("r", encoding="utf-8") as f:
        it = iter(f)
        for raw in it:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            name = Path(parts[9]).name

            R = qvec_to_rotmat(np.array([qw, qx, qy, qz], dtype=np.float64))
            t = np.array([tx, ty, tz], dtype=np.float64)
            C = -R.T @ t
            centers[name] = C
            if return_order:
                order.append(name)

            next(it, None)  # skip points2D line

    if return_order:
        return order, centers
    return centers
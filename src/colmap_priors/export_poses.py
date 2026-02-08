from __future__ import annotations

import argparse
import contextlib
import glob
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .sim3 import apply_sim3, umeyama

# ---------------------------
# Shared small utilities
# ---------------------------


def _iter_images_sorted(image_dir: Path) -> list[str]:
    images = sorted(
        f for f in glob.glob(os.path.join(str(image_dir), "*")) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not images:
        raise RuntimeError(f"No images found in {image_dir}")
    return images


# ---------------------------
# Pi3 exporter (NPZ)
# ---------------------------


def _pi3_build_image_paths(image_dir: Path, interval: int, n_expected: int) -> list[str]:
    """Rebuild image_paths in the same order as pi3.utils.basic.load_images_as_tensor.

    Order is:
      - listdir
      - filter by image extensions
      - sorted
      - take every `interval`-th image
    """
    valid_ext = (".png", ".jpg", ".jpeg")
    filenames = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(valid_ext)])
    if len(filenames) == 0:
        raise ValueError(f"[Pi3] No images found in {image_dir}")

    step = max(int(interval), 1)
    indices = list(range(0, len(filenames), step))
    sel_filenames = [filenames[i] for i in indices]
    image_paths = [str(image_dir / fn) for fn in sel_filenames]

    if len(image_paths) != int(n_expected):
        raise RuntimeError(
            f"[Pi3] Mismatch between tensor batch (N={n_expected}) and "
            f"image_paths from directory (N={len(image_paths)}). "
            f"Check that interval={interval} is the same in all places."
        )
    return image_paths


def export_pi3_to_npz(
    *,
    image_dir: Path,
    output: Path,
    interval: int = 1,
    device: str = "cuda",
    ckpt: str | None = None,
) -> None:
    """Run Pi3 inference on an image directory and export to NPZ.

    Output NPZ keys (unchanged):
      - camera_poses: (N,4,4) c2w
      - image_paths: (N,) strings
    """

    # Lazy imports (optional deps)
    import torch  # type: ignore

    from .external_repos import add_vendor_to_syspath

    add_vendor_to_syspath("PI3")
    from pi3.models.pi3 import Pi3  # type: ignore
    from pi3.utils.basic import load_images_as_tensor  # type: ignore

    def get_device(device_arg: str) -> torch.device:
        if device_arg == "cuda" and not torch.cuda.is_available():
            print("[Pi3] CUDA not available, falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_arg)

    image_dir = Path(image_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    dev = get_device(device)
    print(f"[Pi3] Using device: {dev}")
    print(f"[Pi3] Image dir : {image_dir}")
    print(f"[Pi3] Output npz: {output}")
    print(f"[Pi3] Interval  : {interval}")

    # 1) Load model
    print("[Pi3] Loading model...")
    if ckpt is not None:
        model = Pi3().to(dev).eval()
        print(f"[Pi3] Loading weights from checkpoint: {ckpt}")
        weight = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(weight)
    else:
        print("[Pi3] Using Pi3.from_pretrained('yyfz233/Pi3')")
        model = Pi3.from_pretrained("yyfz233/Pi3").to(dev).eval()

    # 2) Load images as tensor
    print("[Pi3] Loading images with load_images_as_tensor...")
    imgs = load_images_as_tensor(str(image_dir), interval=int(interval))
    if imgs.numel() == 0:
        raise ValueError(f"[Pi3] No images loaded from {image_dir}")

    n = int(imgs.shape[0])
    print(f"[Pi3] Loaded {n} frames -> tensor shape: {tuple(imgs.shape)}")
    imgs = imgs.to(dev)

    # 3) Build image_paths in matching order
    print("[Pi3] Building image_paths list in matching order...")
    image_paths = _pi3_build_image_paths(image_dir, int(interval), n)

    # 4) Inference (no_grad + optional mixed precision)
    print("[Pi3] Running model inference...")
    use_cuda_amp = dev.type == "cuda"
    if use_cuda_amp:
        cap_major, _ = torch.cuda.get_device_capability(dev)
        amp_dtype = torch.bfloat16 if cap_major >= 8 else torch.float16
        print(f"[Pi3] Using CUDA autocast with dtype={amp_dtype}")
    else:
        amp_dtype = None

    with torch.no_grad():
        if use_cuda_amp:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                preds = model(imgs[None])
        else:
            preds = model(imgs[None])

    # 5) Convert only camera_poses to numpy (unchanged)
    print("[Pi3] Converting camera_poses to numpy...")
    out: dict[str, object] = {}
    value = preds["camera_poses"]
    out["camera_poses"] = value.detach().cpu().numpy().squeeze(0)
    out["image_paths"] = np.array(image_paths, dtype=str)

    np.savez(output, **out)
    print(f"[Pi3] Saved predictions to {output}")

    if dev.type == "cuda":
        torch.cuda.empty_cache()
        print("[Pi3] Emptied CUDA cache.")


def main_pi3(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--interval", type=int, default=1)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=("Path to Pi3 checkpoint. If not provided, uses Pi3.from_pretrained()."),
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    export_pi3_to_npz(
        image_dir=Path(args.image_dir),
        output=Path(args.output),
        interval=int(args.interval),
        device=str(args.device),
        ckpt=str(args.ckpt) if args.ckpt is not None else None,
    )
    return 0


# ---------------------------
# DepthAnything3 exporter (NPZ)
# ---------------------------


def export_da3_to_npz(
    *,
    image_dir: Path,
    output: Path,
    model: str = "depth-anything/DA3NESTED-GIANT-LARGE",
    window: int = 10,
    overlap: int = 6,
    stride: int = 0,
    autocast: bool = False,
) -> None:
    """Run DepthAnything3 inference windowed, stitch centers, and export to NPZ.

    Output NPZ keys (unchanged):
      - image_paths: (N,) strings
      - centers: (N,3) float32 (camera centers derived from w2c extrinsics)
      - extrinsics_raw: (N,3,4) float32 (window-local, debug)
      - centers_mode: ["w2c"]

    Conventions:
      - DA3 extrinsics are treated as w2c [R|t].
      - Camera center is C = -R^T t.
      - Window stitching uses Sim(3) from overlap correspondences by index.
    """

    # Lazy imports (optional deps)
    import torch  # type: ignore

    from .external_repos import add_vendor_to_syspath

    add_vendor_to_syspath("DA3")
    from depth_anything_3.api import DepthAnything3  # type: ignore

    image_dir = Path(image_dir)
    output = Path(output)

    window = int(window)
    overlap = int(overlap)
    stride = int(stride) if int(stride) > 0 else max(1, window - overlap)
    if overlap >= window:
        raise ValueError("overlap must be < window")
    if overlap < 3:
        print("[WARN] overlap<3 is risky; use >=4 if possible")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DA3] device={dev}")

    model_obj = DepthAnything3.from_pretrained(model).to(device=dev).eval()
    images = _iter_images_sorted(image_dir)
    n = len(images)
    print(f"[DA3] found {n} images")

    if dev.type == "cuda" and autocast:
        major, _ = torch.cuda.get_device_capability()
        mp_dtype = torch.bfloat16 if major >= 8 else torch.float16
        autocast_ctx: contextlib.AbstractContextManager[None] = torch.amp.autocast("cuda", dtype=mp_dtype)
        print(f"[DA3] autocast dtype={mp_dtype}")
    else:
        autocast_ctx = contextlib.nullcontext()

    centers_global = np.full((n, 3), np.nan, dtype=np.float64)
    extrinsics_raw = np.zeros((n, 3, 4), dtype=np.float32)

    for start in range(0, n, stride):
        end = min(n, start + window)
        batch_paths = images[start:end]
        b = len(batch_paths)
        if b < 2:
            break
        print(f"[DA3] window {start}:{end} (n={b})")

        with torch.no_grad(), autocast_ctx:
            pred = model_obj.inference(batch_paths)

        extr = np.asarray(pred.extrinsics)  # (B,3,4)
        if extr.ndim != 3 or extr.shape[1:] != (3, 4):
            raise RuntimeError(f"Unexpected extrinsics shape: {extr.shape}")

        extrinsics_raw[start:end] = extr.astype(np.float32)

        # w2c [R|t] -> center C = -R^T t
        r = extr[:, :3, :3].astype(np.float64)
        t = extr[:, :3, 3].astype(np.float64)
        c_win = -(np.transpose(r, (0, 2, 1)) @ t[..., None]).squeeze(-1)

        if start == 0:
            centers_global[start:end] = c_win
        else:
            ov = min(overlap, b, start)
            if ov < 3:
                raise RuntimeError(f"[DA3] overlap too small to stitch: ov={ov}")

            x = centers_global[start : start + ov]
            if not np.isfinite(x).all():
                raise RuntimeError("[DA3] Missing global centers in overlap; stitching order broke")

            y = c_win[:ov]
            s, r_sim, t_sim = umeyama(x, y)
            c_glob = apply_sim3(c_win, s, r_sim, t_sim)

            # Keep earlier overlap values
            write_from = ov
            centers_global[start + write_from : end] = c_glob[write_from:, :]

            ov_rmse = np.sqrt(np.mean(np.sum((apply_sim3(y, s, r_sim, t_sim) - x) ** 2, axis=1)))
            print(f"[DA3] stitch overlap rmse={ov_rmse:.4g}  (centers=w2c)")

        if end == n:
            break

        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if not np.isfinite(centers_global).all():
        bad = np.where(~np.isfinite(centers_global).any(axis=1))[0]
        raise RuntimeError(f"[DA3] Missing stitched centers for indices: {bad[:10].tolist()}...")

    out = {
        "image_paths": np.array(images, dtype=object),
        "centers": centers_global.astype(np.float32),
        "extrinsics_raw": extrinsics_raw,
        "centers_mode": np.array(["w2c"], dtype=object),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **out)
    print(f"[DA3] saved {output} centers={out['centers'].shape} mode=w2c")


def main_da3(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--model", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--overlap", type=int, default=6)
    p.add_argument("--stride", type=int, default=0, help="If 0, stride=window-overlap")
    p.add_argument("--autocast", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    export_da3_to_npz(
        image_dir=Path(args.image_dir),
        output=Path(args.output),
        model=str(args.model),
        window=int(args.window),
        overlap=int(args.overlap),
        stride=int(args.stride),
        autocast=bool(args.autocast),
    )
    return 0

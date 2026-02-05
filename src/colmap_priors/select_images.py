from __future__ import annotations

from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def write_image_list(image_dir: Path, *, max_images: int, out_path: Path) -> list[str]:
    image_dir = Path(image_dir)
    out_path = Path(out_path)
    files = sorted([p.name for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

    if max_images <= 0 or max_images >= len(files):
        sel = files
    else:
        # uniform sampling over index range
        if max_images == 1:
            sel = [files[0]]
        else:
            step = (len(files) - 1) / float(max_images - 1)
            idx = [round(i * step) for i in range(max_images)]
            sel = [files[min(int(j), len(files) - 1)] for j in idx]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sel) + "\n", encoding="utf-8")
    return sel
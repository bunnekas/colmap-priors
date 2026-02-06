from __future__ import annotations

import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    # Stream stdout+stderr live (no buffering until process ends)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert proc.stdout is not None
    collected: list[str] = []

    for line in proc.stdout:
        print(line, end="")
        collected.append(line)

    ret = proc.wait()
    if ret != 0:
        out = "".join(collected)
        if "libcudart.so" in out:
            raise RuntimeError(
                "COLMAP failed to start because CUDA runtime libraries were not found "
                "(missing libcudart). Load CUDA module (e.g. `module load cuda/12.8`) "
                "or fix LD_LIBRARY_PATH."
            )
        raise subprocess.CalledProcessError(ret, cmd, output=out)


def database_creator(db: Path, *, colmap_exe: str = "colmap") -> None:
    _run([colmap_exe, "database_creator", "--database_path", str(db)])


def feature_extractor(
    db: Path,
    image_dir: Path,
    *,
    image_list_path: Path | None = None,
    colmap_exe: str = "colmap",
) -> None:
    cmd = [
        colmap_exe,
        "feature_extractor",
        "--database_path",
        str(db),
        "--image_path",
        str(image_dir),
    ]
    if image_list_path is not None:
        cmd += ["--image_list_path", str(image_list_path)]

    try:
        _run(cmd)
    except subprocess.CalledProcessError as e:
        # Some COLMAP builds don't support --image_list_path for feature_extractor.
        # If so, retry without it.
        out = str(getattr(e, "output", "") or "")
        if (
            image_list_path is not None
            and out
            and ("image_list_path" in out or "unrecognized option" in out or "Unknown option" in out)
        ):
            print("[WARN] COLMAP build does not support --image_list_path for feature_extractor; retrying without it.")
            cmd_fallback = [
                colmap_exe,
                "feature_extractor",
                "--database_path",
                str(db),
                "--image_path",
                str(image_dir),
            ]
            _run(cmd_fallback)
            return
        raise


def exhaustive_matcher(db: Path, *, colmap_exe: str = "colmap") -> None:
    _run([colmap_exe, "exhaustive_matcher", "--database_path", str(db)])


def mapper(db: Path, image_dir: Path, out_sparse: Path, *, colmap_exe: str = "colmap") -> None:
    out_sparse.mkdir(parents=True, exist_ok=True)
    _run(
        [
            colmap_exe,
            "mapper",
            "--database_path",
            str(db),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(out_sparse),
        ]
    )


def model_to_txt(model_dir: Path, out_txt: Path, *, colmap_exe: str = "colmap") -> None:
    out_txt.mkdir(parents=True, exist_ok=True)
    _run(
        [
            colmap_exe,
            "model_converter",
            "--input_path",
            str(model_dir),
            "--output_path",
            str(out_txt),
            "--output_type",
            "TXT",
        ]
    )


def pose_prior_mapper(
    db: Path,
    image_dir: Path,
    out_sparse: Path,
    *,
    prior_std: float,
    colmap_exe: str = "colmap",
) -> None:
    out_sparse.mkdir(parents=True, exist_ok=True)
    _run(
        [
            colmap_exe,
            "pose_prior_mapper",
            "--database_path",
            str(db),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(out_sparse),
            "--overwrite_priors_covariance",
            "1",
            "--prior_position_std_x",
            str(prior_std),
            "--prior_position_std_y",
            str(prior_std),
            "--prior_position_std_z",
            str(prior_std),
        ]
    )

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_env(env_path: Path) -> None:
    # Keep scripts usable without installing the package (match other wrappers).
    repo_root = _repo_root()
    src = repo_root / "src"
    sys.path.insert(0, str(src))
    from colmap_priors.config import load_env_file  # type: ignore  # noqa: E402

    load_env_file(env_path)


def _dispatch(python_bin: str, script: Path, passthrough: list[str]) -> None:
    args = list(passthrough)
    if args[:1] == ["--"]:
        args = args[1:]
    subprocess.run([python_bin, str(script), *args], check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Dispatch Pi3/DA3 pose export using configured model interpreters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--env",
        type=Path,
        default=Path("scripts/colmap_priors.env"),
        help="Env file providing PI3_PYTHON / DA3_PYTHON and optional *_REPO paths.",
    )
    ap.add_argument("--pi3-python", default=None, help="Override PI3_PYTHON")
    ap.add_argument("--da3-python", default=None, help="Override DA3_PYTHON")

    sub = ap.add_subparsers(dest="cmd", required=True)
    p_pi3 = sub.add_parser("pi3", help="Run Pi3 exporter")
    p_pi3.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to scripts/export_pi3_poses.py")

    p_da3 = sub.add_parser("da3", help="Run DA3 exporter")
    p_da3.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to scripts/export_da3_poses.py")

    p_both = sub.add_parser("both", help="Run Pi3 then DA3 exporters")
    p_both.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments passed to both exporters (only use options common to both CLIs).",
    )

    args = ap.parse_args(argv)

    _load_env(args.env)

    repo_root = _repo_root()
    scripts_dir = repo_root / "scripts"
    pi3_script = scripts_dir / "export_pi3_poses.py"
    da3_script = scripts_dir / "export_da3_poses.py"

    pi3_python = args.pi3_python or os.environ.get("PI3_PYTHON")
    da3_python = args.da3_python or os.environ.get("DA3_PYTHON")

    if args.cmd in {"pi3", "both"} and not pi3_python:
        raise SystemExit("PI3_PYTHON not set (provide via --pi3-python or env file)")
    if args.cmd in {"da3", "both"} and not da3_python:
        raise SystemExit("DA3_PYTHON not set (provide via --da3-python or env file)")

    if args.cmd == "pi3":
        _dispatch(pi3_python, pi3_script, args.args)
    elif args.cmd == "da3":
        _dispatch(da3_python, da3_script, args.args)
    elif args.cmd == "both":
        _dispatch(pi3_python, pi3_script, args.args)
        _dispatch(da3_python, da3_script, args.args)
    else:
        raise AssertionError(args.cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

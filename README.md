# colmap-priors

<p align="center">
  <img src="examples/courhouse_baseline.png" width="45.9%" />
  <img src="examples/courhouse_da3.png" width="45%" />
</p>
<p align="center"><em>Tanks-and-Temples Courthouse – baseline (left) vs DA3 position priors (right).</em></p>

COLMAP pipeline that injects pose/position priors from [Pi3](https://github.com/yyfz/Pi3) and [DepthAnything3](https://github.com/ByteDance-Seed/Depth-Anything-3) into `pose_prior_mapper`, with Sim(3) alignment against a reference model and trajectory evaluation plots.

## Getting started

Requires COLMAP with `pose_prior_mapper` support, Python >= 3.10, [uv](https://docs.astral.sh/uv/), and optionally [just](https://github.com/casey/just).

```bash
git clone --recurse-submodules https://github.com/bunnekas/colmap_priors.git
cd colmap_priors
uv sync                        # pipeline env (numpy, matplotlib, pyyaml)
just vendor-install             # separate venvs for Pi3 and DA3 under vendor/
```

Copy the config template, point it at your data, and run a scene:

```bash
cp config.yaml config.local.yaml
# edit config.local.yaml — at minimum set data_root and exp_root
just run Courthouse
```

If you cloned without `--recurse-submodules`, run `git submodule update --init --recursive` first.

## Configuration

All settings live in a single YAML file (see [`config.yaml`](config.yaml) for defaults and comments).
The important ones:

| Key | Description |
|---|---|
| `data_root` | Root folder with scene subdirectories (each containing `images/`) |
| `exp_root` | Output directory |
| `prior_position_std` | Position prior std-dev passed to `pose_prior_mapper` |
| `run_pi3` / `run_da3` | Which model branches to run |
| `run_baseline` | Whether to also run vanilla COLMAP mapper |
| `run_plot` | Generate trajectory comparison plots |

Model Python interpreters default to `vendor/<model>/.venv/bin/python` — override with `pi3_python` / `da3_python` if needed.

## What it does

Per scene, the pipeline runs:

1. Feature extraction + exhaustive matching (shared DB)
2. Optional baseline COLMAP reconstruction
3. Model inference (Pi3 poses or DA3 depth-based centers), run in the model's own venv
4. Sim(3) alignment of predictions to a reference model (Umeyama)
5. Prior injection into the COLMAP database
6. `pose_prior_mapper` reconstruction
7. Trajectory plots comparing against the reference

## `just` recipes

```
just sync              install pipeline deps
just vendor-install    set up model venvs (Pi3, DA3)
just run <SCENE>       run full pipeline
just run-batch S1 S2   multiple scenes
just export-pi3 ...    run Pi3 exporter standalone
just export-da3 ...    run DA3 exporter standalone
just lint / fmt        ruff
just test              pytest
just init-config       copy config template
```

Use `just --set config other.yaml run Courthouse` to point at a different config.

## Directory structure

Input data:
```
data_root/<SCENE>/images/       # jpg/png frames
data_root/<SCENE>/sparse/0/     # optional reference model
```

Outputs:
```
exp_root/<SCENE>/
  base.db                       # shared feature/match DB
  colmap_baseline/sparse/0/     # baseline reconstruction (if enabled)
  pi3/                          # predictions, DB, reconstruction
  da3/                          # predictions, DB, reconstruction
  plots/                        # comparison PNGs
```

## Code overview

The pipeline entry point is `src/colmap_priors/run_scene.py`. Model inference lives in `export_poses.py` with lazy torch imports so the main pipeline stays light. Alignment and DB injection go through `align_and_inject.py` which uses `sim3.py` (Umeyama) under the hood. Pi3 and DA3 run in their own venvs via `scripts/export.py`.

The two models are vendored as git submodules under `vendor/`. Each gets its own `.venv` created by `just vendor-install`.

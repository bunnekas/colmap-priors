# colmap_priors – pipeline recipes
# Usage: just <recipe> [args...]
# Requires: just (https://github.com/casey/just)

set dotenv-load := false

# Default config file (override: just --set config ...)
config := "config.local.yaml"

# ── Setup ────────────────────────────────────────────────────────────────────

# Install / sync the lightweight pipeline environment
sync:
    uv sync

# Install with dev tools (ruff, pytest)
sync-dev:
    uv sync --group dev

# Clone / update vendor submodules (Pi3, Depth-Anything-3)
vendor:
    git submodule update --init --recursive

# Create venvs inside vendor submodules and install their deps
vendor-install: vendor
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -f vendor/Pi3/requirements.txt ]; then
        echo "── setting up vendor/Pi3 venv"
        python3 -m venv vendor/Pi3/.venv
        vendor/Pi3/.venv/bin/pip install -r vendor/Pi3/requirements.txt
    fi
    if [ -f vendor/Depth-Anything-3/requirements.txt ]; then
        echo "── setting up vendor/Depth-Anything-3 venv"
        python3 -m venv vendor/Depth-Anything-3/.venv
        vendor/Depth-Anything-3/.venv/bin/pip install -r vendor/Depth-Anything-3/requirements.txt
        vendor/Depth-Anything-3/.venv/bin/pip install -e vendor/Depth-Anything-3
    fi

# ── Pipeline ─────────────────────────────────────────────────────────────────

# Run the full pipeline for a scene
run scene *flags:
    uv run colmap-priors {{ scene }} --config {{ config }} {{ flags }}

# Run pipeline on multiple scenes sequentially
run-batch +scenes:
    #!/usr/bin/env bash
    set -euo pipefail
    for s in {{ scenes }}; do
        echo "━━━ Scene: $s ━━━"
        uv run colmap-priors "$s" --config {{ config }}
    done

# ── Exporters (standalone) ───────────────────────────────────────────────────

# Run Pi3 exporter directly
export-pi3 *args:
    vendor/Pi3/.venv/bin/python scripts/export.py pi3 {{ args }}

# Run DA3 exporter directly
export-da3 *args:
    vendor/Depth-Anything-3/.venv/bin/python scripts/export.py da3 {{ args }}

# ── Quality ──────────────────────────────────────────────────────────────────

# Lint with ruff
lint:
    uv run ruff check src/ scripts/

# Auto-fix lint issues
fix:
    uv run ruff check --fix src/ scripts/

# Format with ruff
fmt:
    uv run ruff format src/ scripts/

# Check formatting without writing
fmt-check:
    uv run ruff format --check src/ scripts/

# Run tests
test *args:
    uv run pytest {{ args }}

# ── Helpers ──────────────────────────────────────────────────────────────────

# Create a local config from the template
init-config:
    @[ -f {{ config }} ] && echo "{{ config }} already exists" || cp config.yaml {{ config }}
    @echo "Edit {{ config }} before running the pipeline."

# Show current config
show-config:
    @echo "config = {{ config }}"
    @[ -f {{ config }} ] && cat {{ config }} || echo "(file not found)"

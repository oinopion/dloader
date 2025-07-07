lint:
    uv run ruff format --check
    uv run ruff check
    just --unstable --format --check

fix:
    uv run ruff format
    uv run ruff check --fix --unsafe-fixes
    just --unstable --format

test *args:
    uv run pytest {{ args }}

typecheck:
    uv run pyright

deps-upgrade:
    uv sync --upgrade

qa: fix typecheck test

build: clean
    uv build

clean:
    rm -rf dist
    rm -rf .pytest_cache
    rm -rf __pycache__
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

release:
    #!/usr/bin/env bash
    set -euo pipefail
    VERSION=$(uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")
    echo "Creating release for version: v$VERSION"
    gh release create "v$VERSION" \
        --title "v$VERSION" \
        --generate-notes \
        --draft

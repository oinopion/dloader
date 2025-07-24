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
    rm -rf dist docs
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

docs-serve:
    uv run pdoc dloader

docs-build:
    uv run pdoc dloader --output-directory docs/

bump-version:
    uv run bump-my-version bump patch

release:
    #!/usr/bin/env bash
    set -euo pipefail
    VERSION=$(uv run bump-my-version show --format json | jq -r '.current_version')
    echo "Creating release for version: v$VERSION"
    gh release create "v$VERSION" \
        --title "v$VERSION" \
        --generate-notes \
        --draft

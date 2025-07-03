lint:
    uv run ruff format --check
    uv run ruff check
    just --unstable --format --check

fix:
    uv run ruff format
    uv run ruff check --fix
    just --unstable --format

test *args:
    uv run pytest {{ args }}

qa: fix lint test

clean:
    rm -rf .pytest_cache
    rm -rf __pycache__
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

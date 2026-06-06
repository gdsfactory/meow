dev:
  uv venv --python 3.12
  uv sync --all-extras
  uv pip install -e .
  uv run pre-commit install

bump version="patch":
    uvx --python 3.12 --from 'bver-cli==0.1.4' bver bump "{{ version }}"

dist:
  uv run python -m build --wheel

uv:
  curl -LsSf https://astral.sh/uv/install.sh | sh

inits:
  cd src/meow/fde && uv run mkinit --nomods --write
  cd src/meow/eme && uv run mkinit --nomods --write
  cd src/meow && uv run mkinit --write
  uv run ruff check --fix || true
  uv run ruff format

ipykernel:
  uv run python -m ipykernel install --user --name meow --display-name meow

test: ipykernel
  uv run pytest -s -n logical

docs:
  sed 's|](docs/|](|g' README.md > docs/index.md
  sed 's|^#|###|g' CHANGELOG.md | sed 's|^### \[0|## [0|g' > docs/changelog.md
  uv run zensical build

serve: docs
  uv run zensical serve

nbrun: ipykernel
  find notebooks -maxdepth 2 -mindepth 1 -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" -not -path "./.venv/*" | xargs parallel -j `nproc --all` uv run papermill {} {} -k meow :::

nbdocs:
  rm -rf docs/nbs/*.ipynb
  find notebooks -maxdepth 1 -mindepth 1 -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" -not -path "./.venv/*" | xargs parallel -j `nproc --all` uv run jupyter nbconvert --to markdown --embed-images {} --output-dir docs/nbs ':::'

nbclean-all:
  find . -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" -not -path "./.venv/*" | xargs just nbclean

nbclean +filenames:
  for filename in {{filenames}}; do \
    uv run --no-sync nbstripout "$filename"; \
    uv run --no-sync nb-clean clean --remove-empty-cells "$filename"; \
    uv run --no-sync jq --indent 1 'del(.metadata.papermill)' "$filename" > "$filename.tmp" && mv "$filename.tmp" "$filename"; \
  done

tree:
  @tree -a -I .git --gitignore

clean:
  find . -name "*.ipynb" | xargs just nbclean
  rm -rf .venv
  rm -rf docs/notebooks/*
  rm -rf site
  rm -rf dist
  find . -name "*.egg_info" | xargs rm -rf
  find . -name "*.so" | xargs rm -rf
  find . -name ".ipynb_checkpoints" | xargs rm -rf
  find . -name ".virtual_documents" | xargs rm -rf
  find . -name ".mypy_cache" | xargs rm -rf
  find . -name ".pytest_cache" | xargs rm -rf
  find . -name ".ruff_cache" | xargs rm -rf
  find . -name "__pycache__" | xargs rm -rf
  find . -name "build" | xargs rm -rf
  find . -name "builds" | xargs rm -rf
  find . -name "modes" | xargs rm -rf
  find src -name "*.c" | xargs rm -rf
  find src -name "*.pyc" | xargs rm -rf
  find src -name "*.pyd" | xargs rm -rf
  find src -name "*.so" | xargs rm -rf

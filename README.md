# ECS 172 Final Project

## Setup with uv

This project uses [uv](https://docs.astral.sh/uv/) for Python environment and dependency management.

### Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# or via Homebrew
brew install uv
```

### Install dependencies

```bash
uv sync
```

This creates a `.venv/` and installs everything from `pyproject.toml` / `uv.lock`.

### Run the project

```bash
uv run main.py
```

`uv run` executes commands inside the project's environment without needing to activate it manually.

### Add a dependency

```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # dev-only dependency
```

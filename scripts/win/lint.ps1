$ErrorActionPreference = 'Stop'
ruff check .
black --check .
mypy src

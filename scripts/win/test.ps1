param(
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

$activate = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    throw "Virtual environment not found at $VenvPath. Run scripts\win\setup.ps1 first."
}

. $activate

pytest -q

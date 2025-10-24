param(
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment at $VenvPath" -ForegroundColor Cyan
    python -m venv $VenvPath
}

$activate = Join-Path $VenvPath "Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    throw "Unable to locate activation script at $activate"
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
. $activate

Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "Installing CPU + dev extras..." -ForegroundColor Cyan
pip install -e ".[cpu,dev]"

Write-Host "Environment ready." -ForegroundColor Green

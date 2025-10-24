param(
    [string]$Output = "dist/colab_bundle.zip"
)

$ErrorActionPreference = "Stop"

$root = Get-Location
$distDir = Join-Path $root "dist"
$stageDir = Join-Path $distDir "colab_bundle"
$zipPath = Join-Path $root $Output

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}
if (Test-Path $stageDir) {
    Remove-Item $stageDir -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $distDir | Out-Null
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

$items = @(
    "README.md",
    "COLAB_README.md",
    "configs",
    "docs",
    "examples",`n    "colab",
    "notebooks/01_colab_starter.md",`n    "notebooks/02_colab_tests.md"
)

foreach ($item in $items) {
    $source = Join-Path $root $item
    if (-not (Test-Path $source)) {
        Write-Warning "Skipping missing item: $item"
        continue
    }

    $destination = Join-Path $stageDir $item
    if ((Get-Item $source).PsIsContainer) {
        Copy-Item $source -Destination $stageDir -Recurse -Force
    } else {
        $destDir = Split-Path $destination -Parent
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Force -Path $destDir | Out-Null
        }
        Copy-Item $source -Destination $destination -Force
    }
}

if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}
Compress-Archive -Path (Join-Path $stageDir '*') -DestinationPath $zipPath -Force
Remove-Item $stageDir -Recurse -Force

Write-Host "Created $zipPath"

param(
    [string]$VenvPath = ".venv",
    [string]$OutDir = ".conan_out",
    [string]$DemoVideo = "examples\tiny_input.mp4"
)

$ErrorActionPreference = "Stop"

$activate = Join-Path $VenvPath "Scripts\Activate.ps1"
if (Test-Path $activate) {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    . $activate
} else {
    Write-Host "Virtual environment not found. Using current Python interpreter." -ForegroundColor Yellow
}

if (-not (Test-Path $DemoVideo)) {
    Write-Host "Creating placeholder demo video at $DemoVideo" -ForegroundColor Cyan
    New-Item -ItemType File -Path $DemoVideo -Force | Out-Null
}

if (Test-Path $OutDir) {
    Remove-Item -Recurse -Force $OutDir
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$steps = @(
    @("python", "-m", "conan_dub.cli.conan_dub", "split", $DemoVideo, "--out", $OutDir, "--dry-run"),
    @("python", "-m", "conan_dub.cli.conan_dub", "diarize", (Join-Path $OutDir "voice.wav"), "--out", (Join-Path $OutDir "diar"), "--dry-run"),
    @("python", "-m", "conan_dub.cli.conan_dub", "align", (Join-Path $OutDir "voice.wav"), "--segments", (Join-Path $OutDir "diar/segments.json"), "--out", (Join-Path $OutDir "aligned.json"), "--dry-run"),
    @("python", "-m", "conan_dub.cli.conan_dub", "frames", $DemoVideo, "--out", (Join-Path $OutDir "frames"), "--dry-run"),
    @("python", "-m", "conan_dub.cli.conan_dub", "detect-faces", (Join-Path $OutDir "frames"), "--out", (Join-Path $OutDir "faces.json"), "--dry-run"),
    @("python", "-m", "conan_dub.cli.conan_dub", "char-seed", (Join-Path $OutDir "frames"), "--out", (Join-Path $OutDir "characters"), "--dry-run"),
    @("python", "-m", "conan_dub.cli.conan_dub", "char-augment", "--chars", (Join-Path $OutDir "characters"), "--faces", (Join-Path $OutDir "faces.json"), "--out", (Join-Path $OutDir "char_aug"), "--dry-run"),
    @("python", "-m", "conan_dub.cli.conan_dub", "ser", (Join-Path $OutDir "voice.wav"), "--aligned", (Join-Path $OutDir "aligned.json"), "--out", (Join-Path $OutDir "emotion.json"), "--dry-run"),
    @("python", "-m", "conan_dub.cli.conan_dub", "fuse", "--aligned", (Join-Path $OutDir "aligned.json"), "--diar", (Join-Path $OutDir "diar/segments.json"), "--ser-path", (Join-Path $OutDir "emotion.json"), "--faces", (Join-Path $OutDir "faces.json"), "--out", (Join-Path $OutDir "manifest.json"), "--dry-run"),
    @("python", "-m", "conan_dub.cli.conan_dub", "translate", (Join-Path $OutDir "manifest.json"), "--out", (Join-Path $OutDir "manifest_it.json"), "--dry-run"),
    @("python", "-m", "conan_dub.cli.conan_dub", "synth", (Join-Path $OutDir "manifest_it.json"), "--out", (Join-Path $OutDir "tts_wavs"), "--dry-run"),
    @("python", "-m", "conan_dub.cli.conan_dub", "mix", (Join-Path $OutDir "tts_wavs"), "--bg", (Join-Path $OutDir "background.wav"), "--out", (Join-Path $OutDir "final_audio.wav"), "--dry-run"),
    @("python", "-m", "conan_dub.cli.conan_dub", "mux", $DemoVideo, "--audio", (Join-Path $OutDir "final_audio.wav"), "--out", (Join-Path $OutDir "output_private.mkv"), "--dry-run")
)

foreach ($cmd in $steps) {
    $commandLine = $cmd -join " "
    Write-Host ">> $commandLine" -ForegroundColor Cyan
    & $cmd[0] @($cmd[1..($cmd.Count - 1)])
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $commandLine"
    }
}

Write-Host "Dry-run demo completed. Outputs located in $OutDir" -ForegroundColor Green

$ErrorActionPreference = 'Stop'

@('.conan_out','build','dist') | ForEach-Object { if (Test-Path ) { Remove-Item -Recurse -Force  } }
Get-ChildItem -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force
Write-Host 'Workspace cleaned.'

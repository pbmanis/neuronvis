# Build the NeuronVis (hocRender GUI) application for Windows with PyInstaller.
# Run from PowerShell:
#     powershell -ExecutionPolicy Bypass -File Installers\build_windows.ps1
# or through ../build_app.sh from Git Bash / MSYS.
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv not found: install it from https://docs.astral.sh/uv/ and run again."
    exit 1
}

# NEURON must be importable in the project environment before it can be bundled.
uv run python -c "import neuron"
if ($LASTEXITCODE -ne 0) {
    Write-Error "NEURON could not be imported in the uv environment; install NEURON for this Python first."
    exit 1
}

Write-Host "Building NeuronVis for Windows ..."
# pyinstaller is added for this run only; it is not a project dependency
uv run --with pyinstaller pyinstaller --noconfirm --clean `
    --distpath Installers/dist --workpath Installers/build `
    Installers/neuronvis.spec
if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller build failed."
    exit $LASTEXITCODE
}

$Exe = Join-Path $RepoRoot "Installers\dist\NeuronVis\NeuronVis.exe"
if (-not (Test-Path $Exe)) {
    Write-Error "Build failed: $Exe was not created."
    exit 1
}

Write-Host ""
Write-Host "Build complete."
Write-Host "Executable: $Exe"
Write-Host "Copy the whole folder $(Split-Path -Parent $Exe) to install it elsewhere."

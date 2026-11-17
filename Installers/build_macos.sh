#!/usr/bin/env bash
# Build the NeuronVis (hocRender GUI) application for macOS with PyInstaller.
# Run from anywhere; normally invoked through ../build_app.sh.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v uv > /dev/null 2>&1; then
    echo "uv not found: install it from https://docs.astral.sh/uv/ and run again." >&2
    exit 1
fi

echo "Building NeuronVis for macOS ..."
# pyinstaller is added for this run only; it is not a project dependency
uv run --with pyinstaller pyinstaller --noconfirm --clean \
    --distpath Installers/dist --workpath Installers/build \
    Installers/neuronvis.spec

app="$repo_root/Installers/dist/NeuronVis.app"
if [ ! -d "$app" ]; then
    echo "Build failed: $app was not created." >&2
    exit 1
fi

echo
echo "Build complete."
echo "Application: $app"
echo "Executable : $app/Contents/MacOS/NeuronVis"
echo "Start it with: open \"$app\"   (or drag it to /Applications)"
echo "The app is not code-signed: on another Mac, right-click it and choose Open the first time."

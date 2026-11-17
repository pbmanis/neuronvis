#!/usr/bin/env bash
# Build the NeuronVis (hocRender GUI) application for Linux with PyInstaller.
# Run from anywhere; normally invoked through ../build_app.sh.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v uv > /dev/null 2>&1; then
    echo "uv not found: install it from https://docs.astral.sh/uv/ and run again." >&2
    exit 1
fi

echo "Building NeuronVis for Linux ..."
# pyinstaller is added for this run only; it is not a project dependency
uv run --with pyinstaller pyinstaller --noconfirm --clean \
    --distpath Installers/dist --workpath Installers/build \
    Installers/neuronvis.spec

exe="$repo_root/Installers/dist/NeuronVis/NeuronVis"
if [ ! -x "$exe" ]; then
    echo "Build failed: $exe was not created." >&2
    exit 1
fi

echo
echo "Build complete."
echo "Executable: $exe"
echo "Copy the whole folder $repo_root/Installers/dist/NeuronVis to install it elsewhere."
echo "Qt 6 needs the system xcb libraries at run time (e.g. Debian/Ubuntu: sudo apt install libxcb-cursor0)."

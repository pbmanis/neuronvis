#!/usr/bin/env bash
# Build the NeuronVis (hocRender GUI) application for the current OS.
# The platform build scripts and the PyInstaller spec are in Installers/;
# the result is written to Installers/dist.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$(uname -s)" in
    Darwin)
        exec bash "$repo_root/Installers/build_macos.sh" "$@"
        ;;
    Linux)
        exec bash "$repo_root/Installers/build_linux.sh" "$@"
        ;;
    MINGW* | MSYS* | CYGWIN*)
        exec powershell.exe -NoProfile -ExecutionPolicy Bypass \
            -File "$repo_root/Installers/build_windows.ps1" "$@"
        ;;
    *)
        echo "Unsupported operating system: $(uname -s)" >&2
        echo 'On Windows without bash, run Installers\build_windows.ps1 from PowerShell.' >&2
        exit 1
        ;;
esac

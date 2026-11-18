#!/usr/bin/env bash
# Build PyGAMer (GAMer2 mesh smoothing) into the project's .venv, for
# hocRenderGUI's Mesh > Smoothing = GAMer. Optional: Taubin smoothing needs
# nothing extra, and pygamer is not a project dependency.
#
# PyGAMer 2.0.7 is source-only on PyPI and does not build as published with
# current tools. This script downloads it to a temporary directory and:
#   - uses pybind11 v2.13.6 instead of the pinned v2.7.1, which does not
#     compile against Python >= 3.11 (PyFrameObject became opaque);
#   - sets CMAKE_POLICY_VERSION_MINIMUM=3.5, as CMake 4 rejects the bundled
#     TetGen's old cmake_minimum_required;
#   - on macOS, compiles with Homebrew LLVM 18: Apple clang >= 19 rejects the
#     template-template matching in GAMer's CASC library, and the flag that
#     restored the old rule was removed in clang 21. LLVM 18 needs an older
#     deployment target and the SDK's library path to link.
# Run from anywhere: Installers/build_pygamer.sh
set -euo pipefail

version="2.0.7"
pybind11_tag="v2.13.6"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="$repo_root/.venv"

if ! command -v uv > /dev/null 2>&1; then
    echo "uv not found: install it from https://docs.astral.sh/uv/ and run again." >&2
    exit 1
fi
if [ ! -x "$venv/bin/python" ]; then
    echo "No environment at $venv: run 'uv sync' in $repo_root first." >&2
    exit 1
fi
if ! command -v cmake > /dev/null 2>&1; then
    echo "cmake not found (macOS: brew install cmake; Linux: your package manager)." >&2
    exit 1
fi

work="$(mktemp -d -t neuronvis_pygamer)"  # left in place (the system clears its temporary directory)
echo "Downloading pygamer $version source to $work ..."
url="$("$venv/bin/python" -c "
import json, urllib.request
d = json.load(urllib.request.urlopen('https://pypi.org/pypi/pygamer/$version/json'))
print([u['url'] for u in d['urls'] if u['packagetype'] == 'sdist'][0])
")"
curl -sSL -o "$work/pygamer.tar.gz" "$url"
tar xzf "$work/pygamer.tar.gz" -C "$work"
src="$work/pygamer-$version"

echo "Using pybind11 $pybind11_tag (pinned: v2.7.1)"
sed -i.orig "s/GIT_TAG[[:space:]]*v2\.7\.1/GIT_TAG         $pybind11_tag/" "$src/pygamer/CMakeLists.txt"
grep -q "$pybind11_tag" "$src/pygamer/CMakeLists.txt" || { echo "Could not patch the pybind11 version." >&2; exit 1; }

export CMAKE_POLICY_VERSION_MINIMUM=3.5
case "$(uname -s)" in
    Darwin)
        llvm="$(brew --prefix llvm@18 2> /dev/null || true)"
        if [ -z "$llvm" ] || [ ! -x "$llvm/bin/clang++" ]; then
            echo "Homebrew LLVM 18 not found: brew install llvm@18, then run again." >&2
            exit 1
        fi
        sdk="$(xcrun --show-sdk-path)"
        export CC="$llvm/bin/clang" CXX="$llvm/bin/clang++" SDKROOT="$sdk"
        export MACOSX_DEPLOYMENT_TARGET=14.0
        export LDFLAGS="-L$sdk/usr/lib -Wl,-syslibroot,$sdk"
        echo "Compiler: $("$CXX" --version | head -1)"
        ;;
    Linux)
        echo "Compiler: $(c++ --version | head -1) (if it fails in CASC templates, try an older g++/clang++ via CC/CXX)"
        ;;
    *)
        echo "Unsupported system $(uname -s): build PyGAMer by hand (see the comments in this script)." >&2
        exit 1
        ;;
esac

echo "Building and installing into $venv (several minutes) ..."
VIRTUAL_ENV="$venv" uv pip install --no-cache "$src"

"$venv/bin/python" -c "import pygamer; print('pygamer imported from', pygamer.__file__)"
echo
echo "Done. Restart hocRenderGUI; Mesh > Smoothing = GAMer is now available."
echo "The downloaded source is in $work (not needed any more)."
echo "Note: 'uv sync' removes packages that are not project dependencies; use 'uv sync --inexact' to keep pygamer."

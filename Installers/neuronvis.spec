# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the neuronvis hocRender GUI.

Build with one of the platform scripts in this directory (or ../build_app.sh),
which run, from the repository root:

    uv run --with pyinstaller pyinstaller --noconfirm --clean \
        --distpath Installers/dist --workpath Installers/build Installers/neuronvis.spec

Output (one-folder build):
    macOS   : Installers/dist/NeuronVis.app
    Linux   : Installers/dist/NeuronVis/NeuronVis
    Windows : Installers/dist/NeuronVis/NeuronVis.exe
"""

import sys
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

app_name = "NeuronVis"
spec_dir = Path(SPECPATH)
repo_root = spec_dir.parent


def not_tests(name: str) -> bool:
    return ".tests" not in name and ".examples" not in name and ".crxd" not in name


# NEURON: neuron/__init__.py sets NEURONHOME to neuron/.data/share/nrn, which
# holds the hoc library (stdlib.hoc, ...), and libnrniv then dlopen()s
# libnrnpython from $NEURONHOME/../../lib. Binary dependency analysis moves the
# NEURON libraries to the top level of the bundle, so put the ones loaded at
# run time back in neuron/.data/lib (MPI, CoreNEURON and the X11 GUI are not used).
neuron_lib = Path(find_spec("neuron").origin).parent / ".data" / "lib"
neuron_runtime_libs = ("libnrniv", "libnrnpython", "libpywrapper", "librxdmath")
binaries = [
    (str(lib), "neuron/.data/lib")
    for lib in sorted(neuron_lib.glob("*"))
    if lib.name.startswith(neuron_runtime_libs) and lib.suffix in (".dylib", ".so", ".dll")
]
datas = collect_data_files("neuron", subdir=".data/share")
datas += collect_data_files("vispy")  # glsl shaders, fonts
datas += collect_data_files("pyqtgraph", excludes=["**/examples/**"])  # icons, colormaps
datas += collect_data_files("matplotlib")

hiddenimports = collect_submodules("neuronvis")
hiddenimports += collect_submodules("neuron", filter=not_tests)
hiddenimports += collect_submodules("pyqtgraph", filter=not_tests)
hiddenimports += collect_submodules("OpenGL.platform") + collect_submodules("OpenGL.arrays")
hiddenimports += collect_submodules("pylibrary")
hiddenimports += [
    "vispy.app.backends._pyqt6",  # vispy picks its backend at run time
    "matplotlib.backends.backend_qtagg",
    "mpl_toolkits.mplot3d",
]

a = Analysis(
    [str(spec_dir / "hocRenderGUI_app.py")],
    pathex=[str(repo_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # only one Qt binding may be bundled; the GUI uses PyQt6.
    # setuptools is not needed at run time, and PyInstaller's setuptools hook
    # fails with the importlib-metadata 1.6 backport pinned in pyproject.toml.
    excludes=[
        "PyQt5",
        "PySide2",
        "PySide6",
        "tkinter",
        "IPython",
        "jupyter",
        "setuptools",
        "pkg_resources",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI application: no terminal window on Windows/macOS
    argv_emulation=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=app_name,
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{app_name}.app",
        icon=None,
        bundle_identifier=None,
        version=version("neuronvis"),
        info_plist={
            "NSHighResolutionCapable": True,
            "CFBundleDocumentTypes": [
                {
                    "CFBundleTypeName": "Neuron morphology",
                    "CFBundleTypeExtensions": ["swc", "hoc", "hocx"],
                    "CFBundleTypeRole": "Viewer",
                }
            ],
        },
    )

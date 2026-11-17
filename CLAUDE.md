# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`neuronvis` is a Python library and CLI toolkit for visualizing neuron morphology files (`.hoc`, `.hocx`, `.swc`) from the NEURON simulation environment. It supports coloring by section type, ion channel density, or animated membrane voltage from simulation results.

## Environment & Installation

The project uses **uv** for dependency management (Python ≥ 3.13.7). The active virtual environment is `.venv`.

```bash
# Install/sync dependencies
uv sync

# Activate environment
source .venv/bin/activate

# Install in editable mode (if needed)
pip install -e .
```

The legacy `nvis_venv` (Python 3.11, pip-based) is superseded by `.venv`.


## Coding conventions
- Never execute git commands.
- Never delete a file or directory.
- Always ask before making any change to a source file, and make changes incrementally (ask for every change, even if they are related).
- When fixing a bug, comment out the original line(s) rather than deleting,
  and add a comment `# Claude fixed YYYY-MM-DD: <reason>` on the replacement.
- `CP("color", message)` is the project's preferred coloured print utility. if "CP.
   cprint" or cprint is encountered, change the import to use the CP() format.

## Entry Points / CLI Commands

Three scripts are registered in `pyproject.toml`:

```bash
# Render a morphology file (primary tool)
hocRender <file.swc|file.hoc|file.hocx> [options]

# Convert SWC to HOC format
swc_to_hoc <file.swc> [options]

# Image viewer (TIF files)
hocViewer
```

### hocRender example
```bash
hocRender VCN_Rostral_P60_Granule_Cell_Node_List_06-202500000.swc \
  --secmap sbem3 -r pyqtgraph -s cylinders \
  -p new_counting_points.csv -m sec-type \
  --sx 0.024 --sy 0.024 --sz 0.07
```

### swc_to_hoc example
```bash
swc_to_hoc Granule_Cell_1_um_v200000.swc -s sbem3 --center -v
```

There are no automated tests in this repository.

## Architecture

### Data flow

```
.swc file
    └─► SWC (swc_to_hoc.py)          # parse, sort topology, prune
            └─► write_hoc() → .hocx  # also generates .segmap sidecar
                    │
.hoc/.hocx file     │
    └─► HocReader (hoc_reader.py)     # wraps NEURON h object
            ├─ sections / sec_groups
            ├─ mechanisms
            └─ get_geometry() → vertexes, edges
                    │
                    └─► HocViewer (hoc_viewer.py)   # rendering backends
                                └─► hoc_graphics.py  # GL primitives
```

`Render` in `hocRender.py` is the top-level orchestrator that wires these together and handles CLI arguments.

### Key classes

| Class | File | Role |
|---|---|---|
| `SWC` | `swc_to_hoc.py` | Parse SWC files, topological sort, prune branches, emit HOC |
| `HocReader` | `hoc_reader.py` | Wrap NEURON `h` object; enumerate sections, mechanisms, geometry |
| `HocViewer` | `hoc_viewer.py` | Rendering backends (pyqtgraph/OpenGL, matplotlib, vispy) |
| `Render` | `hocRender.py` | CLI orchestrator; reads file, instantiates HocReader + HocViewer |

### File formats

| Extension | Description |
|---|---|
| `.swc` | Standard morphology: `n T x y z R P` per node |
| `.hoc` | NEURON HOC script defining sections |
| `.hocx` | HOC with `// seg=N` annotations linking each `pt3dadd` to original SWC node |
| `.segmap` | Sidecar to `.hocx`: maps `section[N]: swc_id, ...` for reverse lookup |

`HocReader` accepts all three input types; for `.swc` it calls `SWC` internally and generates HOC in memory without writing a file.

### Section type mappings (`swc_sectypes.py`)

The `section_map` parameter selects which integer→name mapping to use. Pass it to both `SWC` and `HocReader`/`Render`:

| Map name | Use case |
|---|---|
| `swc` | Standard SWC types (soma/axon/dendrite) |
| `sbem` | SBEM bushy cell extended types |
| `sbem2` | SyGlass SBEM from May 2021 |
| `sbem3` | Granule cell / node list data (Spirou 2024) |
| `sbem7` | Draft expanded types (Spirou 2026, not in CLI choices yet) |

`idsofpart_*` dicts in the same file define which type IDs belong to "dendrite", "axon", "soma", etc. for pruning operations.

Colors for each section-type name are in `renderer_colormaps.py` (`section_colors` dict). When adding a new type to a `*_sectypes` dict, add a corresponding color entry there.

### Display options

- **Renderers** (`-r`): `pyqtgraph` (default, OpenGL), `mpl` (matplotlib), `vispy`
- **Styles** (`-s`): `cylinders` (default), `graph`, `volume`, `surface`
- **Modes** (`-m`): `sec-type` (color by type), `mechanism` (color by channel density), `vm` (animate Vm from `.p` simulation file)

### Coordinate scaling

SWC files are expected in micrometers. If the source data is in pixels, pass `--sx`/`--sy`/`--sz` (and optionally `--sr` for radius) to `hocRender`, or `--radiiscale` to `swc_to_hoc`. The `--center` flag translates the first node to the origin.

### External dependency

`pylibrary` is a private package installed from GitHub (`https://github.com/pbmanis/pylibrary`). It provides `fileselector` for native file dialogs. It is declared as a `uv` source in `pyproject.toml`.

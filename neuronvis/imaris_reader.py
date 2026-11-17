"""
imaris_reader : read filament tracings from Imaris (.ims) files and convert
them to SWC.

An .ims file is HDF5. Each filament object made in Imaris is stored as
/Scene8/Content/FilamentsN, with
    Name attribute : the object name shown in Imaris ("Filaments 1 Basal", ...)
    Vertex         : PositionX, PositionY, PositionZ, Radius (um)
    Edge           : VertexA, VertexB (indices into the filament's vertices)
    Filament       : one row per filament (tree) in the object, with its root
                     vertex (the "beginning point"; at the soma its radius is
                     the soma radius) and its vertex and edge index ranges.
Only the first time point is read. Image data in the file is never loaded.

SWC types are taken from the object name: "Apical" -> 4 (apical dendrite),
"Basal" or "Dendrite" -> 3 (basal dendrite), "Axon" -> 2, anything else -> 3;
root vertices are written as soma (1). When several objects are converted
together, roots at the same place (within the soma radius) share one soma
node, so a cell traced as separate "Basal" and "Apical" objects becomes one
tree. Pieces that are not connected to a root are left out by default (an SWC
for NEURON must be one connected tree); with connected_only=False they are
written as separate SWC trees (parent -1). The result is checked with
topology_check.check_swc.

Command line:
    ims_to_swc file.ims [-o out.swc] [--objects "Filaments 1 Basal" ...] [--list] [--keep-fragments]

Requires h5py.
"""

import argparse
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order

from neuronvis import topology_check

content_path = "Scene8/Content"

soma_type = 1
type_keywords = [  # first match in the (lower case) object name wins
    ("apical", 4),
    ("basal", 3),
    ("dendrite", 3),
    ("axon", 2),
]
default_type = 3


@dataclass
class FilamentObject:
    """one Imaris filament object; may hold several filaments (trees)"""

    name: str  # object name in Imaris
    group: str  # HDF5 group name, e.g. "Filaments1"
    positions: np.ndarray  # (N, 3) um
    radii: np.ndarray  # (N,) um
    edges: np.ndarray  # (E, 2) indices into positions
    roots: list = field(default_factory=list)  # root (beginning point) vertex of each filament


def _open(filename: Union[str, Path]):
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Reading Imaris .ims files requires h5py (pip install h5py)") from exc
    return h5py.File(filename, "r")


def _attribute_string(group, name: str) -> str:
    """Imaris stores string attributes as arrays of single bytes"""
    value = group.attrs.get(name)
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode()
    return b"".join(value).decode()


def _filament_groups(hf) -> list:
    if content_path not in hf:
        return []
    content = hf[content_path]
    groups = [name for name in content.keys() if name.startswith("Filaments")]
    return sorted(groups, key=lambda name: int(name[len("Filaments") :] or 0))


def list_filament_objects(filename: Union[str, Path]) -> list:
    """names of the filament objects in an .ims file, made unique if needed"""
    with _open(filename) as hf:
        names = [
            _attribute_string(hf[content_path][group], "Name") or group for group in _filament_groups(hf)
        ]
    return _unique(names)


def _unique(names: list) -> list:
    unique = []
    for name in names:
        label, k = name, 2
        while label in unique:
            label = f"{name} ({k:d})"
            k += 1
        unique.append(label)
    return unique


def read_filament_objects(filename: Union[str, Path], names: Union[list, None] = None) -> list:
    """
    Read the filament objects of an .ims file (all, or those in *names*, as
    returned by list_filament_objects). Returns a list of FilamentObject.
    """
    objects = []
    with _open(filename) as hf:
        groups = _filament_groups(hf)
        labels = _unique([_attribute_string(hf[content_path][g], "Name") or g for g in groups])
        for group, label in zip(groups, labels):
            if names is not None and label not in names:
                continue
            g = hf[content_path][group]
            vertex = g["Vertex"][:]
            edge = g["Edge"][:]
            positions = np.column_stack([vertex["PositionX"], vertex["PositionY"], vertex["PositionZ"]])
            radii = np.asarray(vertex["Radius"], dtype=float)
            edges = np.empty((0, 2), dtype=np.int64)
            roots = []
            for filament in g["Filament"][:]:
                if filament["ID_Time"] != 0:  # first time point only
                    continue
                # vertex indices in Edge and IndexVertexRoot are relative to the
                # filament's first vertex
                v0 = filament["IndexVertexBegin"]
                e_range = slice(filament["IndexEdgeBegin"], filament["IndexEdgeEnd"])
                e = np.column_stack([edge["VertexA"][e_range], edge["VertexB"][e_range]]) + v0
                if len(e) > 0 and (e.min() < v0 or e.max() >= filament["IndexVertexEnd"]):
                    raise ValueError(f"{group:s}: edge indices outside the filament's vertices")
                edges = np.vstack([edges, e])
                roots.append(int(v0 + filament["IndexVertexRoot"]))
            objects.append(
                FilamentObject(
                    name=label,
                    group=group,
                    positions=positions.astype(float),
                    radii=radii,
                    edges=edges,
                    roots=roots,
                )
            )
    return objects


def swc_type_for(name: str) -> int:
    lower = name.lower()
    for keyword, swc_type in type_keywords:
        if keyword in lower:
            return swc_type
    return default_type


def filaments_to_swc(objects: list, connected_only: bool = False) -> tuple:
    """
    Convert filament objects to an SWC array with columns
    id, type, x, y, z, radius, parent (ids from 1; parents precede children).

    Returns (swc, n_fragments, n_fragment_points): pieces not connected to a
    root are counted, and left out if connected_only.
    """
    rows = []
    somas = []  # (swc id, position, radius) of the soma nodes written so far
    n_fragments = 0
    n_fragment_points = 0
    for obj in objects:
        n = len(obj.positions)
        swc_type = swc_type_for(obj.name)
        graph = coo_matrix(
            (np.ones(len(obj.edges)), (obj.edges[:, 0], obj.edges[:, 1])), shape=(n, n)
        ).tocsr()
        swc_id = np.zeros(n, dtype=np.int64)  # 0 = not written yet
        starts = [(root, True) for root in obj.roots]
        starts += [(v, False) for v in range(n)]  # anything left is a fragment
        for start, is_root in starts:
            if swc_id[start] != 0:
                continue
            order, predecessors = breadth_first_order(graph, start, directed=False, return_predecessors=True)
            if not is_root:
                n_fragments += 1
                n_fragment_points += len(order)
                if connected_only:
                    swc_id[order] = -1  # mark as handled
                    continue
            for v in order:
                if v == start and is_root:
                    shared = _find_soma(somas, obj.positions[v])
                    if shared is not None:  # same soma as an earlier object
                        swc_id[v] = shared
                        continue
                    new_id = len(rows) + 1
                    rows.append([new_id, soma_type, *obj.positions[v], obj.radii[v], -1])
                    somas.append((new_id, obj.positions[v], obj.radii[v]))
                    swc_id[v] = new_id
                    continue
                parent = -1 if v == start else swc_id[predecessors[v]]
                new_id = len(rows) + 1
                rows.append([new_id, swc_type, *obj.positions[v], obj.radii[v], parent])
                swc_id[v] = new_id
    swc = np.array(rows, dtype=float).reshape(-1, 7)
    return swc, n_fragments, n_fragment_points


def _find_soma(somas: list, position: np.ndarray) -> Union[int, None]:
    for swc_id, soma_position, soma_radius in somas:
        if np.linalg.norm(position - soma_position) <= max(soma_radius, 1.0):
            return swc_id
    return None


def write_swc(swc: np.ndarray, filename: Union[str, Path], source: Union[str, Path], objects: list) -> None:
    header = [
        "Converted from Imaris filaments by neuronvis.imaris_reader",
        f"Source file: {str(source):s}",
        f"Filament objects: {', '.join(obj.name for obj in objects):s}",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'):s}",
        "Units: um",
        "id type x y z radius parent",
    ]
    np.savetxt(filename, swc, fmt="%d %d %.4f %.4f %.4f %.4f %d", header="\n".join(header))


def default_objects(names: list) -> list:
    """
    Objects to convert when none are chosen: those whose names give them a
    type (Basal, Apical, ...), since a cell is often traced both whole and as
    typed parts; otherwise all of them.
    """
    typed = [name for name in names if any(keyword in name.lower() for keyword, _ in type_keywords)]
    return typed if len(typed) > 0 else list(names)


def ims_to_swc(
    filename: Union[str, Path],
    swc_file: Union[str, Path, None] = None,
    names: Union[list, None] = None,
    connected_only: bool = True,
) -> tuple:
    """
    Convert the filament objects *names* (default: see default_objects) of an
    .ims file to *swc_file* (default: next to the .ims file).
    Returns (swc_file, n_fragments, n_fragment_points, topology report).
    """
    filename = Path(filename)
    if names is None:
        names = default_objects(list_filament_objects(filename))
    objects = read_filament_objects(filename, names)
    if len(objects) == 0:
        raise ValueError(f"No filament objects found in {filename.name:s}")
    swc, n_fragments, n_fragment_points = filaments_to_swc(objects, connected_only=connected_only)
    if swc_file is None:
        swc_file = filename.with_suffix(".swc")
    write_swc(swc, swc_file, filename, objects)
    report = topology_check.check_swc(swc, source=Path(swc_file).name)
    return Path(swc_file), n_fragments, n_fragment_points, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Imaris filament tracings (.ims) to SWC")
    parser.add_argument("ims_file", help="Imaris .ims file")
    parser.add_argument("-o", "--output", default=None, help="SWC file (default: next to the .ims file)")
    parser.add_argument(
        "--objects",
        nargs="+",
        default=None,
        help="filament objects to convert (default: the typed ones, e.g. Basal and Apical, else all)",
    )
    parser.add_argument("--list", action="store_true", help="list the filament objects and exit")
    parser.add_argument(
        "--keep-fragments",
        action="store_true",
        help="also write pieces that are not connected to the soma, as separate trees "
        "(the SWC is then not one connected tree)",
    )
    args = parser.parse_args()

    names = list_filament_objects(args.ims_file)
    if args.list or len(names) == 0:
        print(f"Filament objects in {Path(args.ims_file).name:s}:")
        for name in names:
            print(f"    {name:s}  (SWC type {swc_type_for(name):d})")
        if len(names) == 0:
            print("    (none)")
        return
    swc_file, n_fragments, n_points, report = ims_to_swc(
        args.ims_file, args.output, args.objects, connected_only=not args.keep_fragments
    )
    print(f"Wrote {str(swc_file):s}")
    if n_fragments > 0:
        action = "written as separate trees" if args.keep_fragments else "left out"
        print(f"{n_fragments:d} piece(s) ({n_points:d} points) not connected to the soma were {action:s}")
    print(report)
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

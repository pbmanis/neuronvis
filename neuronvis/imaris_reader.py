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
tree. A piece not connected to a root is joined to the tree when one of its
ends lies on a tree point (see filaments_to_swc). Pieces that still are not
connected are left out by default (an SWC for NEURON must be one connected
tree); with connected_only=False they are written as separate SWC trees
(parent -1). The result is checked with topology_check.check_swc.

Command line:
    ims_to_swc file.ims [-o out.swc] [--objects "Filaments 1 Basal" ...] [--list] [--keep-fragments] [--no-join]

Requires h5py.
"""

import argparse
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

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


def filaments_to_swc(objects: list, connected_only: bool = True, join_tolerance: float = 1e-3) -> tuple:
    """
    Convert filament objects to an SWC array with columns
    id, type, x, y, z, radius, parent (ids from 1; parents precede children,
    and nodes are in depth-first order, so each unbranched run is listed
    consecutively, as NEURON's Import3D expects).

    1. The trees that grow from each object's roots are written (objects with
       roots at the same soma share one soma node).
    2. A piece not connected to a root is joined to the tree when one of its
       ends lies on a tree point (within *join_tolerance* um; 0 turns joining
       off). Every point of the piece that coincides with a tree point is taken
       to be that point (Imaris objects can trace the same stretch twice), and
       the rest of the piece is attached where the overlap ends. This repeats,
       so that pieces can join pieces joined before them.
    3. Pieces still not connected are left out if connected_only, or else
       written as separate trees.

    Returns (swc, info) with info = {"joins": [description, ...],
    "fragments": count, "fragment_points": count, "loops": edges left out
    because joining closed a loop}.
    """
    rows = []
    somas = []  # (swc id, position, radius) of the soma nodes written so far
    info = {"joins": [], "fragments": 0, "fragment_points": 0, "loops": 0}
    ids = []  # per object: SWC id of each vertex (0: not written)
    neighbors = []
    for obj in objects:
        n = len(obj.positions)
        graph = coo_matrix((np.ones(len(obj.edges)), (obj.edges[:, 0], obj.edges[:, 1])), shape=(n, n))
        graph = (graph + graph.T).tocsr()
        neighbors.append([graph.indices[graph.indptr[v] : graph.indptr[v + 1]] for v in range(n)])
        ids.append(np.zeros(n, dtype=np.int64))

    def write(oi: int, start: int, parent_id: int, allowed: Union[np.ndarray, None] = None) -> None:
        """depth-first from *start* through unwritten (and *allowed*) vertices of object *oi*"""
        obj, swc_type = objects[oi], swc_type_for(objects[oi].name)
        stack = [(start, parent_id)]
        while stack:
            v, pid = stack.pop()
            if ids[oi][v] != 0:
                info["loops"] += 1  # reached twice: an edge that would close a loop
                continue
            ids[oi][v] = len(rows) + 1
            rows.append([ids[oi][v], swc_type, *obj.positions[v], obj.radii[v], pid])
            for nb in neighbors[oi][v][::-1]:  # reversed, so the first neighbour is written first
                if ids[oi][nb] == 0 and (allowed is None or allowed[nb]):
                    stack.append((nb, ids[oi][v]))

    # 1. trees from the roots
    for oi, obj in enumerate(objects):
        for root in obj.roots:
            if ids[oi][root] != 0:
                continue
            shared = _find_soma(somas, obj.positions[root])
            if shared is None:
                shared = len(rows) + 1
                rows.append([shared, soma_type, *obj.positions[root], obj.radii[root], -1])
                somas.append((shared, obj.positions[root], obj.radii[root]))
            ids[oi][root] = shared
            for nb in neighbors[oi][root][::-1]:
                if ids[oi][nb] == 0:
                    write(oi, nb, shared)

    # 2. join pieces whose end lies on the tree
    pieces = _unwritten_pieces(ids, neighbors)
    joined = True
    while joined and join_tolerance > 0 and len(pieces) > 0:
        joined = False
        tree = cKDTree(np.array(rows)[:, 2:5])
        for oi, members in list(pieces):
            obj = objects[oi]
            distance, nearest = tree.query(obj.positions[members])
            on_tree = distance <= join_tolerance
            ends = np.array([len(neighbors[oi][v]) <= 1 for v in members])
            if not np.any(on_tree & ends):
                continue
            for v, row in zip(members[on_tree], nearest[on_tree]):
                ids[oi][v] = int(rows[row][0])  # the piece's point is that tree point
            rest = np.zeros(len(obj.positions), dtype=bool)
            rest[members[~on_tree]] = True
            attached = []
            for c in members[on_tree]:  # attach the rest where the overlap ends
                for nb in neighbors[oi][c]:
                    if rest[nb] and ids[oi][nb] == 0:
                        attached.append(rows[ids[oi][c] - 1][2:5])
                        before = len(rows)
                        write(oi, nb, int(ids[oi][c]), allowed=rest)
                        attached[-1] = (attached[-1], len(rows) - before, rows[ids[oi][c] - 1][1])
            info["joins"].append(_join_description(obj.name, int(on_tree.sum()), attached))
            joined = True
        pieces = _unwritten_pieces(ids, neighbors)  # what is left (parts of pieces not reached)

    # 3. anything still not connected
    for oi, members in pieces:
        info["fragments"] += 1
        info["fragment_points"] += len(members)
        if not connected_only:
            write(oi, int(members[0]), -1)
    swc = np.array(rows, dtype=float).reshape(-1, 7)
    return swc, info


def _unwritten_pieces(ids: list, neighbors: list) -> list:
    """(object index, vertex array) for each connected group of unwritten vertices"""
    pieces = []
    for oi in range(len(ids)):
        seen = ids[oi] != 0
        for v in np.flatnonzero(~seen):
            if seen[v]:
                continue
            members, stack = [], [v]
            seen[v] = True
            while stack:
                u = stack.pop()
                members.append(u)
                for nb in neighbors[oi][u]:
                    if not seen[nb]:
                        seen[nb] = True
                        stack.append(nb)
            pieces.append((oi, np.array(sorted(members))))
    return pieces


type_names = {1: "soma", 2: "axon", 3: "basal dendrite", 4: "apical dendrite"}


def _article(word: str) -> str:
    return ("an " if word[:1] in "aeiou" else "a ") + word


def _join_description(name: str, n_overlap: int, attached: list) -> str:
    parts = [
        f"{n:d} points at ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) on {_article(type_names.get(int(t), f'type {int(t)}'))} point"
        for xyz, n, t in attached
    ]
    text = f"joined a piece of '{name}' to the tree: " + ("; ".join(parts) if parts else "no new points")
    if n_overlap > 1:
        text += f" ({n_overlap:d} points coinciding with the tree were taken as those tree points)"
    return text


def _find_soma(somas: list, position: np.ndarray) -> Union[int, None]:
    for swc_id, soma_position, soma_radius in somas:
        if np.linalg.norm(position - soma_position) <= max(soma_radius, 1.0):
            return swc_id
    return None


def write_swc(
    swc: np.ndarray,
    filename: Union[str, Path],
    source: Union[str, Path],
    objects: list,
    info: Union[dict, None] = None,
    connected_only: bool = True,
    join_tolerance: float = 1e-3,
) -> None:
    """write the SWC with a header recording the source and everything done to the data"""
    header = provenance(source, objects, info, connected_only, join_tolerance) + ["id type x y z radius parent"]
    np.savetxt(filename, swc, fmt="%d %d %.4f %.4f %.4f %.4f %d", header="\n".join(header))


def provenance(
    source: Union[str, Path],
    objects: list,
    info: Union[dict, None] = None,
    connected_only: bool = True,
    join_tolerance: float = 1e-3,
) -> list:
    """lines describing the source and the conversion (without comment markers)"""
    lines = [
        "Converted from Imaris filaments by neuronvis.imaris_reader",
        f"Source file: {str(source):s}",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'):s}",
        "Units: um, as stored in the .ims file (no scaling, no translation); first time point only",
        "Filament objects and the SWC type given to each (from the object name):",
    ]
    lines += [f"  {obj.name:s} ({obj.group:s}): type {swc_type_for(obj.name):d}" for obj in objects]
    lines += [
        "Conversion:",
        "  root (beginning) points written as soma (type 1); roots within the soma radius share one soma node",
        "  nodes in depth-first order (each unbranched run is consecutive)",
    ]
    if join_tolerance > 0:
        lines.append(f"  pieces whose end lies on the tree (within {join_tolerance:g} um) joined to it:")
        joins = [] if info is None else info["joins"]
        lines += [f"    {j:s}" for j in joins] or ["    none needed"]
    else:
        lines.append("  pieces not connected to a root were not joined (joining off)")
    if info is not None and info["fragments"] > 0:
        what = "left out" if connected_only else "written as separate trees (parent -1)"
        lines.append(
            f"  {info['fragments']:d} piece(s), {info['fragment_points']:d} points, not connected to the soma: {what:s}"
        )
    if info is not None and info["loops"] > 0:
        lines.append(f"  {info['loops']:d} edge(s) left out because they would have closed a loop")
    return lines


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
    join_tolerance: float = 1e-3,
) -> tuple:
    """
    Convert the filament objects *names* (default: see default_objects) of an
    .ims file to *swc_file* (default: next to the .ims file).
    Returns (swc_file, info, topology report); info as from filaments_to_swc.
    The joins made are listed among the report's warnings.
    """
    filename = Path(filename)
    if names is None:
        names = default_objects(list_filament_objects(filename))
    objects = read_filament_objects(filename, names)
    if len(objects) == 0:
        raise ValueError(f"No filament objects found in {filename.name:s}")
    swc, info = filaments_to_swc(objects, connected_only=connected_only, join_tolerance=join_tolerance)
    if swc_file is None:
        swc_file = filename.with_suffix(".swc")
    write_swc(swc, swc_file, filename, objects, info, connected_only, join_tolerance)
    info["provenance"] = provenance(filename, objects, info, connected_only, join_tolerance)
    report = topology_check.check_swc(swc, source=Path(swc_file).name)
    report.warnings = [f"repaired: {j:s}" for j in info["joins"]] + report.warnings
    if info["loops"] > 0:
        report.warnings.append(f"{info['loops']:d} edge(s) left out because joining would have closed a loop")
    return Path(swc_file), info, report


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
    parser.add_argument(
        "--no-join",
        action="store_true",
        help="do not join pieces whose end lies on the tree",
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
    swc_file, info, report = ims_to_swc(
        args.ims_file,
        args.output,
        args.objects,
        connected_only=not args.keep_fragments,
        join_tolerance=0.0 if args.no_join else 1e-3,
    )
    print(f"Wrote {str(swc_file):s}")
    if info["fragments"] > 0:
        action = "written as separate trees" if args.keep_fragments else "left out"
        print(
            f"{info['fragments']:d} piece(s) ({info['fragment_points']:d} points) "
            f"not connected to the soma were {action:s}"
        )
    print(report)
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

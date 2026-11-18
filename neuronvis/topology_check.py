"""
topology_check : verify that a morphology is one connected tree that NEURON
can build.

Two levels are checked:

check_swc(data)
    SWC nodes (a file, an (N, 7) array, or the structured array used by
    swc_to_hoc.SWC). Errors: duplicate ids, parents that do not exist, nodes
    that are their own parent, anything other than exactly one root (each
    extra root is a separate, disconnected tree; swc_to_hoc converts only the
    first), and cycles (nodes that cannot be reached from a root).
    Warnings: parents listed after their children, radius <= 0, child points
    at the same position as their parent, and points at the same position as
    other, unrelated points (the same stretch traced twice).

check_sections(h)
    Sections after the morphology is loaded into NEURON. Every section must be
    connected to at least one other section (a terminal section has one
    connection, its parent; a continuing section has two, parent and child;
    a branch point three or more), there must be exactly one root section, and
    every section must be reachable from it. NEURON itself refuses to connect
    sections in a loop, so cycles cannot occur here.
    Warnings (children at the middle of a one-point soma drawn as a cylinder
    are expected, as in NEURON's Import3D): sections without 3D points, with a single 3D point (NEURON
    gives them zero length; a one-point soma needs to be expanded, as
    NEURON's Import3D does) or with zero length, children
    attached partway along their parent (x not 0 or 1) or by their 1 end, and
    gaps between a child's first 3D point and the point where it attaches to
    its parent (beyond the parent's largest radius, so that dendrites leaving
    the surface of a soma are not counted).

Both return a TopologyReport; str(report) gives a readable summary and
report.ok is False when there are errors. report.markers locates the problems
for display: for each piece not connected to the main tree, its point nearest
the main tree ("disconnected"), and for each gap at a connection, the child's
end ("gap"). report.fragments holds the line segments of pieces that are not
connected (check_swc only: swc_to_hoc does not convert them, so they are not
otherwise drawn). Marker and fragment coordinates are in the file's own
coordinates for check_swc and in NEURON's for check_sections.
"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from scipy.spatial import cKDTree

max_listed = 10  # items listed per problem before "... and N more"


@dataclass
class TopologyReport:
    source: str
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    markers: list = field(default_factory=list)  # Marker
    fragments: list = field(default_factory=list)  # (N, 2, 3) arrays of line segments

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def __str__(self) -> str:
        state = "OK" if self.ok else f"{len(self.errors):d} error(s)"
        lines = [f"Topology check of {self.source:s}: {state:s}"]
        if len(self.summary) > 0:
            lines.append("    " + ", ".join(f"{k:s}: {v!s}" for k, v in self.summary.items()))
        lines += [f"  ERROR: {e:s}" for e in self.errors]
        lines += [f"  warning: {w:s}" for w in self.warnings]
        return "\n".join(lines)


@dataclass
class Marker:
    kind: str  # "disconnected" (error) or "gap" (warning)
    end: np.ndarray  # xyz of the end to point at
    toward: np.ndarray  # xyz of the nearest point it should connect to
    label: str
    section: str = ""  # for "gap": the child section


def _nearest_pair(piece: np.ndarray, main: Union[cKDTree, None]) -> tuple:
    """point of *piece* nearest the main tree, that main tree point, and their distance"""
    if main is None:
        return piece[0], piece[0], 0.0
    distances, nearest = main.query(piece)
    k = int(np.argmin(distances))
    return piece[k], main.data[nearest[k]], float(distances[k])


def _listed(items: list) -> str:
    items = [str(i) for i in items]
    text = ", ".join(items[:max_listed])
    if len(items) > max_listed:
        text += f", ... and {len(items) - max_listed:d} more"
    return text


def _swc_columns(data: Union[str, Path, np.ndarray]) -> tuple:
    """(ids, types, xyz, radii, parents) from a file or array"""
    if isinstance(data, (str, Path)):
        data = np.loadtxt(data, comments="#", ndmin=2)
    if data.dtype.names is not None:  # swc_to_hoc.SWC structured array
        xyz = np.column_stack([data["x"], data["y"], data["z"]])
        return data["id"].astype(np.int64), data["type"].astype(int), xyz, data["r"], data["parent"].astype(np.int64)
    data = np.asarray(data, dtype=float).reshape(-1, 7)
    return data[:, 0].astype(np.int64), data[:, 1].astype(int), data[:, 2:5], data[:, 5], data[:, 6].astype(np.int64)


def check_swc(data: Union[str, Path, np.ndarray], source: Union[str, None] = None) -> TopologyReport:
    """check the node-level tree of SWC data (see module docstring)"""
    if source is None:
        source = Path(data).name if isinstance(data, (str, Path)) else "SWC data"
    report = TopologyReport(source=source)
    ids, types, xyz, radii, parents = _swc_columns(data)
    n = len(ids)
    report.summary["nodes"] = n
    if n == 0:
        report.errors.append("no nodes")
        return report

    duplicates = [i for i, count in Counter(ids.tolist()).items() if count > 1]
    if len(duplicates) > 0:
        report.errors.append(f"{len(duplicates):d} duplicate node id(s): {_listed(duplicates)}")
    index = {node_id: k for k, node_id in enumerate(ids.tolist())}

    self_parent = ids[parents == ids]
    if len(self_parent) > 0:
        report.errors.append(f"{len(self_parent):d} node(s) are their own parent: {_listed(self_parent)}")
    is_root = parents < 0
    missing = [(i, p) for i, p in zip(ids[~is_root], parents[~is_root]) if p not in index]
    if len(missing) > 0:
        report.errors.append(
            f"{len(missing):d} node(s) have a parent that does not exist (node: parent): "
            + _listed([f"{i}: {p}" for i, p in missing])
        )

    # walk down from every root
    children = {}
    for k in np.flatnonzero(~is_root):
        children.setdefault(int(parents[k]), []).append(k)
    tree_of = np.full(n, -1)
    tree_sizes = []
    for t, root in enumerate(np.flatnonzero(is_root)):
        stack = [root]
        tree_of[root] = t
        size = 0
        while stack:
            k = stack.pop()
            size += 1
            for c in children.get(int(ids[k]), []):
                if tree_of[c] < 0:
                    tree_of[c] = t
                    stack.append(c)
        tree_sizes.append(size)
    roots = ids[is_root]
    report.summary["trees"] = len(roots)
    if len(roots) == 0:
        report.errors.append("no root node (parent -1): the nodes form a cycle")
    elif len(roots) > 1:
        detail = [f"{r} ({s:d} nodes)" for r, s in zip(roots, tree_sizes)]
        report.errors.append(
            f"{len(roots):d} separate trees (roots: {_listed(detail)}); only one connected tree is "
            f"allowed, and swc_to_hoc converts only the first ({tree_sizes[0]:d} of {n:d} nodes)"
        )
    unreachable = ids[(tree_of < 0) & np.isin(parents, ids)]  # parent exists, yet not reachable: a cycle
    if len(unreachable) > 0:
        report.errors.append(f"{len(unreachable):d} node(s) are in a cycle: {_listed(unreachable)}")

    # locate the pieces that are not part of the first tree (the one swc_to_hoc converts)
    parent_index = np.array([index.get(int(p), -1) for p in parents])
    in_main = tree_of == 0
    main_tree = cKDTree(xyz[in_main]) if np.any(in_main) else None
    pieces = [(f"tree from node {roots[t]}", tree_of == t) for t in range(1, len(roots))]
    if np.any(tree_of < 0):
        pieces.append(("nodes in a cycle or with a missing parent", tree_of < 0))
    for label, members in pieces:
        end, toward, distance = _nearest_pair(xyz[members], main_tree)
        report.markers.append(
            Marker("disconnected", end, toward, f"{label} ({int(members.sum()):d} nodes), {distance:.2f} um from the main tree")
        )
        k = np.flatnonzero(members & (parent_index >= 0))
        if len(k) > 0:
            report.fragments.append(np.stack([xyz[k], xyz[parent_index[k]]], axis=1))

    # warnings
    has_parent = parent_index >= 0
    late = ids[has_parent & (parent_index > np.arange(n))]
    if len(late) > 0:
        report.warnings.append(f"{len(late):d} node(s) are listed before their parent (swc_to_hoc reorders them)")
    bad_radius = ids[radii <= 0]
    if len(bad_radius) > 0:
        report.warnings.append(f"{len(bad_radius):d} node(s) with radius <= 0: {_listed(bad_radius)}")
    k = np.flatnonzero(has_parent)
    same_place = ids[k[np.all(xyz[k] == xyz[parent_index[k]], axis=1)]]
    if len(same_place) > 0:
        report.warnings.append(
            f"{len(same_place):d} node(s) at the same position as their parent (zero-length segments): "
            + _listed(same_place)
        )
    duplicates = [
        (ids[a], ids[b])
        for a, b in sorted(cKDTree(xyz).query_pairs(1e-3))
        if parent_index[a] != b and parent_index[b] != a
    ]
    if len(duplicates) > 0:
        report.warnings.append(
            f"{len(duplicates):d} pair(s) of unrelated nodes at the same position (traced twice?): "
            + _listed([f"{a}={b}" for a, b in duplicates])
        )
    return report


def check_sections(h: object = None, source: str = "NEURON sections", gap_tolerance: float = 1.0) -> TopologyReport:
    """
    check the section-level tree currently loaded in NEURON (see module
    docstring). *gap_tolerance* (um) is added to the parent's largest radius
    when looking for gaps at connections.
    """
    if h is None:
        from neuron import h
    report = TopologyReport(source=source)
    # not define_shape(): it moves child sections to close gaps, hiding them
    sections = list(h.allsec())
    n = len(sections)
    report.summary["sections"] = n
    if n == 0:
        report.errors.append("no sections")
        return report

    roots = []
    isolated = []
    connections = Counter()  # number of connections -> number of sections
    mid_attached = []
    reversed_attached = []
    gaps = []
    no_points = []
    one_point = []
    zero_length = []
    for sec in sections:
        parent_seg = sec.parentseg()
        n_children = len(sec.children())
        n_connections = n_children + (0 if parent_seg is None else 1)
        connections[n_connections] += 1
        if parent_seg is None:
            roots.append(sec)
            if n_children == 0 and n > 1:
                isolated.append(sec.name())
        if sec.n3d() == 0:
            no_points.append(sec.name())
        elif sec.n3d() == 1:
            one_point.append(sec.name())
        elif sec.L <= 0:
            zero_length.append(sec.name())
        if parent_seg is None:
            continue
        x = parent_seg.x
        if 0 < x < 1 and not (x == 0.5 and is_soma_cylinder(parent_seg.sec)):  # soma cylinder: middle is right
            mid_attached.append(f"{sec.name()} at {parent_seg.sec.name()}({x:.3g})")
        if sec.orientation() != 0:
            reversed_attached.append(sec.name())
        gap = _connection_gap(sec, parent_seg)
        if gap is not None and gap[0] > _max_radius(parent_seg.sec) + gap_tolerance:
            gaps.append((gap[0], sec.name(), parent_seg.sec.name()))
            report.markers.append(
                Marker(
                    "gap", gap[1], gap[2], f"{sec.name()} -> {parent_seg.sec.name()}: {gap[0]:.2f} um gap", sec.name()
                )
            )

    report.summary["root"] = roots[0].name() if len(roots) == 1 else f"{len(roots):d} roots"
    report.summary["terminal (1 connection)"] = connections[1]
    report.summary["continuing (2)"] = connections[2]
    report.summary["branching (3+)"] = sum(v for k, v in connections.items() if k >= 3)

    if len(isolated) > 0:
        report.errors.append(f"{len(isolated):d} section(s) not connected to any other section: {_listed(isolated)}")
    if len(roots) > 1:
        trees = [f"{r.name()} ({_subtree_size(r):d} sections)" for r in roots]
        report.errors.append(f"{len(roots):d} separate trees (root sections: {_listed(trees)}); only one is allowed")
        # point at each smaller tree's end nearest the largest one
        roots.sort(key=_subtree_size, reverse=True)
        main_points = _tree_points(roots[0])
        main_tree = cKDTree(main_points) if len(main_points) > 0 else None
        for r in roots[1:]:
            points = _tree_points(r)
            if len(points) == 0:
                continue
            end, toward, distance = _nearest_pair(points, main_tree)
            report.markers.append(
                Marker(
                    "disconnected",
                    end,
                    toward,
                    f"tree from {r.name()} ({_subtree_size(r):d} sections), {distance:.2f} um from the main tree",
                )
            )
    if len(roots) == 1:
        reached = _subtree_size(roots[0])
        if reached != n:
            report.errors.append(f"only {reached:d} of {n:d} sections can be reached from the root {roots[0].name()}")

    if len(no_points) > 0:
        report.warnings.append(f"{len(no_points):d} section(s) without 3D points: {_listed(no_points)}")
    if len(one_point) > 0:
        report.warnings.append(
            f"{len(one_point):d} section(s) with a single 3D point, so zero length and no membrane: "
            + _listed(one_point)
        )
    if len(zero_length) > 0:
        report.warnings.append(f"{len(zero_length):d} section(s) of zero length: {_listed(zero_length)}")
    if len(mid_attached) > 0:
        report.warnings.append(
            f"{len(mid_attached):d} section(s) attached partway along their parent: {_listed(mid_attached)}"
        )
    if len(reversed_attached) > 0:
        report.warnings.append(
            f"{len(reversed_attached):d} section(s) attached by their 1 end: {_listed(reversed_attached)}"
        )
    if len(gaps) > 0:
        gaps.sort(reverse=True)
        report.warnings.append(
            f"{len(gaps):d} connection(s) with a gap between child and parent (largest first): "
            + _listed([f"{c} -> {p}: {g:.2f} um" for g, c, p in gaps])
            + "; NEURON's define_shape() closes them by moving each child and everything beyond it"
        )
    return report


def _subtree_size(sec: object) -> int:
    size = 0
    stack = [sec]
    while stack:
        s = stack.pop()
        size += 1
        stack.extend(s.children())
    return size


def _points(sec: object) -> np.ndarray:
    return np.array([[sec.x3d(i), sec.y3d(i), sec.z3d(i)] for i in range(sec.n3d())]).reshape(-1, 3)


def _tree_points(sec: object) -> np.ndarray:
    """3D points of a section and everything below it"""
    points = []
    stack = [sec]
    while stack:
        s = stack.pop()
        points.append(_points(s))
        stack.extend(s.children())
    return np.concatenate(points, axis=0)


def is_soma_cylinder(sec: object) -> bool:
    """
    True for a one-point soma written as a 3-point cylinder along x with L = diam
    (swc_to_hoc, like NEURON's Import3D)
    """
    if sec.n3d() != 3:
        return False
    d = [sec.diam3d(i) for i in range(3)]
    x = [sec.x3d(i) for i in range(3)]
    same_yz = all(np.isclose(sec.y3d(i), sec.y3d(1)) and np.isclose(sec.z3d(i), sec.z3d(1)) for i in (0, 2))
    return bool(
        same_yz
        and np.allclose(d, d[1])
        and np.isclose(x[1] - x[0], d[1] / 2.0, atol=1e-3)
        and np.isclose(x[2] - x[1], d[1] / 2.0, atol=1e-3)
    )


def _max_radius(sec: object) -> float:
    return max((sec.diam3d(i) for i in range(sec.n3d())), default=0.0) / 2.0


def _connection_gap(sec: object, parent_seg: object) -> Union[tuple, None]:
    """
    (distance, child end xyz, parent xyz) from the child's attached end to the
    parent's 3D location at the connection
    """
    parent = parent_seg.sec
    if sec.n3d() == 0 or parent.n3d() == 0:
        return None
    end = 0 if sec.orientation() == 0 else sec.n3d() - 1
    child_point = np.array([sec.x3d(end), sec.y3d(end), sec.z3d(end)])
    arc = np.array([parent.arc3d(i) for i in range(parent.n3d())])
    pts = _points(parent)
    target = parent_seg.x * parent.L
    parent_point = np.array([np.interp(target, arc, pts[:, j]) for j in range(3)])
    return float(np.linalg.norm(child_point - parent_point)), child_point, parent_point


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Check that a morphology is one connected tree NEURON can build")
    parser.add_argument("files", nargs="+", help=".swc, .hoc or .hocx files")
    parser.add_argument("--secmap", default="swc", help="section map for .swc files (default: swc)")
    args = parser.parse_args()
    failed = False
    for filename in args.files:
        report = check_file(filename, section_map=args.secmap)
        print(report)
        failed = failed or not report.ok
    raise SystemExit(1 if failed else 0)


def check_file(filename: Union[str, Path], section_map: str = "swc") -> TopologyReport:
    """
    check one file: an .swc file is checked as nodes and then, after
    conversion, as NEURON sections; a .hoc/.hocx file as NEURON sections.
    Uses a fresh set of NEURON sections (deletes any loaded ones).
    """
    import neuron

    from neuronvis import hoc_reader

    filename = Path(filename).resolve()
    neuron.h("forall delete_section()")
    if filename.suffix == ".swc":
        node_report = check_swc(filename)
        if not node_report.ok:
            return node_report  # sections built from a broken tree would only repeat the errors
    hoc_reader.HocReader(str(filename), section_map=section_map)
    report = check_sections(neuron.h, source=filename.name)
    if filename.suffix == ".swc":
        report.summary = {"nodes": node_report.summary["nodes"], **report.summary}
        report.warnings = node_report.warnings + report.warnings
    return report


if __name__ == "__main__":
    main()

"""
mesh_tools : clean and smooth triangle meshes (e.g. SyGlass .obj exports).

merge_seams()   merge vertices duplicated at the seams between the blocks a
                mesh was exported in, so that the surface is one connected
                piece where it should be (the area does not change).
taubin_smooth() lambda/mu smoothing (Taubin 1995): removes the voxel
                "staircase" while keeping the volume. numpy/scipy only.
gamer_smooth()  GAMer2's smooth() (PyGAMer): weighted vertex smoothing and
                edge flipping, which improves the triangles' shape but moves
                vertices less than a voxel. PyGAMer is optional and must be
                built (see Installers/build_pygamer.sh); it runs here in a
                separate process (see main) because it can take minutes.
write_obj()     write an OBJ with a header describing what was done.

Run as a script for GAMer:
    python -m neuronvis.mesh_tools gamer input.obj output.obj --iterations 6
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Union

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components

taubin_lambda = 0.5
taubin_mu = -0.53


def surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    v = np.asarray(vertices, dtype=np.float64)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def enclosed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """volume enclosed by a closed surface (absolute value; meaningful only when closed)"""
    v = np.asarray(vertices, dtype=np.float64)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0))


def edge_counts(faces: np.ndarray, n_vertices: int) -> tuple:
    """numbers of boundary (1 face) and non-manifold (>2 faces) edges"""
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1).astype(np.int64)
    _, counts = np.unique(e[:, 0] * (n_vertices + 1) + e[:, 1], return_counts=True)
    return int(np.sum(counts == 1)), int(np.sum(counts > 2))


def pieces(vertices: np.ndarray, faces: np.ndarray) -> tuple:
    """(face label of the connected piece each face belongs to, area of each piece)"""
    n = len(vertices)
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    _, label = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)
    face_label = label[faces[:, 0]]
    v = np.asarray(vertices, dtype=np.float64)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    face_area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return face_label, np.bincount(face_label, weights=face_area)


def merge_seams(vertices: np.ndarray, faces: np.ndarray, decimals: int = 5) -> tuple:
    """
    Merge vertices at the same position (to *decimals* places in the file's
    units), drop the faces that this makes degenerate, and the vertices no
    face uses. Returns (vertices, faces, info, lines); *lines* describe what
    was done, for the info panel, the Report and file headers.
    """
    n_before, f_before = len(vertices), len(faces)
    boundary_before, _ = edge_counts(faces, n_before)
    _, inverse = np.unique(np.round(vertices.astype(np.float64), decimals), axis=0, return_inverse=True)
    inverse = inverse.ravel()  # numpy 2 returns it as a column with axis=0
    first = np.unique(inverse, return_index=True)[1]  # one original vertex for each merged position
    f = inverse[faces]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[keep]
    used, f = np.unique(f, return_inverse=True)
    f = f.reshape(-1, 3).astype(np.int32)
    v = vertices[first][used]
    boundary, nonmanifold = edge_counts(f, len(v))
    face_label, piece_area = pieces(v, f)
    piece_area = piece_area[np.unique(face_label)]
    total = float(piece_area.sum())
    info = {
        "merged": n_before - len(v),
        "degenerate": int(f_before - len(f)),
        "boundary_before": boundary_before,
        "boundary": boundary,
        "nonmanifold": nonmanifold,
        "pieces": len(piece_area),
        "largest_fraction": float(piece_area.max() / total) if total > 0 else 0.0,
        "small_pieces": int(np.sum(piece_area < 1.0)),
        "small_area": float(piece_area[piece_area < 1.0].sum()),
    }
    lines = [
        f"seams merged: {info['merged']:d} duplicated vertices joined (positions equal to {decimals:d} decimals)"
        + (f", {info['degenerate']:d} degenerate faces dropped" if info["degenerate"] else ""),
        f"  open edges {boundary_before:d} -> {boundary:d}; non-manifold edges {nonmanifold:d}",
        f"  {info['pieces']:d} connected piece(s); the largest has {100 * info['largest_fraction']:.1f}% of the area"
        + (
            f"; {info['small_pieces']:d} under 1 um2 ({info['small_area']:.2f} um2 in all)"
            if info["small_pieces"]
            else ""
        ),
    ]
    return v, f, info, lines


def taubin_smooth(vertices: np.ndarray, faces: np.ndarray, iterations: int) -> np.ndarray:
    """
    Taubin lambda/mu smoothing with the umbrella (neighbour mean) operator:
    each iteration a shrinking step (lambda) and an inflating step (mu), so
    that roughness is removed but the volume is kept. Topology is unchanged.
    """
    n = len(vertices)
    rows = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]])
    cols = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]])
    adjacency = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    adjacency.data[:] = 1.0  # each neighbour once
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    mean_of_neighbours = diags(1.0 / np.maximum(degree, 1.0)) @ adjacency
    v = vertices.astype(np.float64)
    for _ in range(iterations):
        v = v + taubin_lambda * (mean_of_neighbours @ v - v)
        v = v + taubin_mu * (mean_of_neighbours @ v - v)
    return v.astype(np.float32)


def gamer_available() -> tuple:
    """(True, "") if PyGAMer can be imported, else (False, how to get it)"""
    if importlib.util.find_spec("pygamer") is not None:
        return True, ""
    return False, (
        "PyGAMer is not installed. It has no wheels for this Python; build it into the "
        "environment with Installers/build_pygamer.sh (see the comments in that script)."
    )


def gamer_smooth(vertices: np.ndarray, faces: np.ndarray, iterations: int, log=print, progress=None) -> tuple:
    """
    GAMer2 smooth(max_iter=iterations, preserve_ridges=False) on all vertices.
    The mesh must be manifold and orientable. Returns (vertices, faces); GAMer
    flips edges, so the faces change too.
    progress(done, total, text) is called as the steps finish: the setup, then
    each iteration. smooth() is called one iteration at a time to allow that;
    this is the same as one call with max_iter=iterations (each iteration
    smooths, recomputes the normals and flips edges; a call only adds one more
    normal computation at its start).
    """
    total = iterations + 1
    if progress is None:

        def progress(done: int, total: int, text: str) -> None:
            pass
    import tempfile

    import pygamer

    # PyGAMer reads meshes from files; this one is overwritten on each call
    work = Path(tempfile.gettempdir(), "neuronvis_gamer")
    work.mkdir(exist_ok=True)
    src = Path(work, "gamer_input.obj")
    write_obj(src, vertices, faces, [])
    mesh = pygamer.readOBJ(str(src))
    components, orientable, manifold = mesh.compute_orientation()
    log(f"GAMer: {components:d} component(s), orientable {orientable!s}, manifold {manifold!s}")
    if not (orientable and manifold):
        raise ValueError("GAMer needs a manifold, orientable surface (merge the seams first?)")
    for vertex in mesh.vertexIDs:  # GAMer only moves selected vertices
        vertex.data().selected = True
    progress(1, total, "mesh ready")
    t0 = time.time()
    # mesh.smooth(max_iter=iterations, preserve_ridges=False)
    for i in range(iterations):
        mesh.smooth(max_iter=1, preserve_ridges=False)
        progress(i + 2, total, f"iteration {i + 1:d} of {iterations:d} ({time.time() - t0:.0f} s)")
    log(f"GAMer: smooth({iterations:d}) took {time.time() - t0:.0f} s")
    v, _, f = mesh.to_ndarray()
    return v.astype(np.float32), f.astype(np.int32)


def obj_header(filename: Union[str, Path], max_lines: int = 40) -> list:
    """the comment lines at the start of an OBJ file, without the '#'"""
    lines = []
    with open(filename, "r", errors="replace") as fh:
        for line in fh:
            if not line.startswith("#") or len(lines) >= max_lines:
                break
            if line[1:].strip():
                lines.append(line[1:].strip())
    return lines


def write_obj(filename: Union[str, Path], vertices: np.ndarray, faces: np.ndarray, header: list) -> None:
    with open(filename, "w") as fh:
        for line in header:
            fh.write(f"# {line:s}\n")
        np.savetxt(fh, vertices, fmt="v %.6f %.6f %.6f")
        np.savetxt(fh, np.asarray(faces) + 1, fmt="f %d %d %d")


def main() -> None:
    """command line, used by the GUI to run GAMer in its own process"""
    from neuronvis import obj_reader

    parser = argparse.ArgumentParser(description="Smooth a triangle mesh (.obj)")
    parser.add_argument("method", choices=["gamer", "taubin"])
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--iterations", type=int, default=6)
    args = parser.parse_args()
    vertices, faces = obj_reader.read_obj(args.input)
    area = surface_area(vertices, faces)
    print(f"read {args.input:s}: {len(vertices):d} vertices, {len(faces):d} faces, area {area:.2f}", flush=True)
    if args.method == "gamer":
        ok, message = gamer_available()
        if not ok:
            sys.exit(message)
        # "progress done total text" lines are read by hocRenderGUI's progress bar
        vertices, faces = gamer_smooth(
            vertices,
            faces,
            args.iterations,
            log=lambda s: print(s, flush=True),
            progress=lambda done, total, text: print(f"progress {done:d} {total:d} {text:s}", flush=True),
        )
        method = f"GAMer2 smooth(max_iter={args.iterations:d}, preserve_ridges=False), all vertices selected"
    else:
        vertices = taubin_smooth(vertices, faces, args.iterations)
        method = f"Taubin lambda {taubin_lambda:g}, mu {taubin_mu:g}, {args.iterations:d} iterations"
    smoothed = surface_area(vertices, faces)
    print(f"smoothed: area {smoothed:.2f} ({smoothed / area:.4f} of before)", flush=True)
    write_obj(args.output, vertices, faces, [f"smoothed from {args.input:s}", f"smoothing: {method:s}"])


if __name__ == "__main__":
    main()

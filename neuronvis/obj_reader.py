"""
obj_reader : read triangle meshes from Wavefront .obj files (for example,
meshes exported by SyGlass).

Only geometry is read: vertices ("v x y z") and faces ("f i j k ..."). Face
indices may be written as i, i/t, i//n or i/t/n, may be negative (relative to
the end of the vertex list), and polygons with more than three vertices are
split into triangles. Normals, texture coordinates, groups and materials are
ignored.

SyGlass writes coordinates in physical units (um), the same space as the swc
reconstructions, so a mesh can be drawn on top of a morphology.
"""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def read_obj(filename: Union[str, Path]) -> tuple:
    """
    Read the vertices and triangular faces of an .obj file.

    Returns
    -------
    vertices : (N, 3) float32 array
    faces : (M, 3) int32 array of 0-based indices into vertices
    """
    try:
        vertices, faces = _read_triangles_fast(filename)
    except (ValueError, pd.errors.ParserError):
        vertices, faces = _read_general(filename)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"No mesh (vertices and faces) found in {str(filename):s}")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"Face indices out of range in {str(filename):s}")
    return vertices, faces


def _read_triangles_fast(filename: Union[str, Path]) -> tuple:
    """
    Fast path for files with only "v x y z" and "f i j k" (plain integer)
    records, as SyGlass writes them. Raises ValueError or ParserError for
    anything else, so that the caller can fall back to the general reader.
    """
    table = pd.read_csv(
        filename,
        sep=" ",
        comment="#",
        header=None,
        names=["record", "a", "b", "c"],
        dtype={"record": str},
        skip_blank_lines=True,
        engine="c",
    )
    vertices = table.loc[table.record == "v", ["a", "b", "c"]].to_numpy(dtype=np.float32)
    face_table = table.loc[table.record == "f", ["a", "b", "c"]]
    if face_table.isna().any().any():  # polygon with fewer than 3 vertices
        raise ValueError("incomplete face")
    faces = face_table.to_numpy(dtype=np.int64)  # "1/2/3" style indices raise ValueError
    return vertices, _zero_based(faces, len(vertices))


def _read_general(filename: Union[str, Path]) -> tuple:
    """line by line reader for any .obj geometry"""
    vertices = []
    faces = []
    with open(filename, "r") as fh:
        for line in fh:
            fields = line.split()
            if len(fields) == 0:
                continue
            if fields[0] == "v":
                vertices.append([float(x) for x in fields[1:4]])
            elif fields[0] == "f":
                corners = [int(field.split("/")[0]) for field in fields[1:]]
                for k in range(1, len(corners) - 1):  # fan triangulation of polygons
                    faces.append([corners[0], corners[k], corners[k + 1]])
    vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.array(faces, dtype=np.int64).reshape(-1, 3)
    return vertices, _zero_based(faces, len(vertices))


def _zero_based(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """.obj indices start at 1; negative indices count back from the last vertex"""
    faces = np.where(faces < 0, faces + n_vertices, faces - 1)
    return faces.astype(np.int32)

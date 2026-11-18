"""
section_picker : find the rendered segment under the mouse.

The morphology is drawn as truncated cones between consecutive 3D points
(the edges of HocReader.get_geometry). A ray from the camera through the mouse
position is tested against all of them at once, each treated as a capsule
whose radius is the cone's radius at the point of closest approach plus a
few pixels of tolerance (so that thin processes can be picked). Of the
segments the ray passes through, the one whose surface it meets first, i.e.
the one nearest the viewer, is picked, so parts hidden behind others are
never chosen.
"""

from typing import Union

import numpy as np


class SegmentPicker:
    def __init__(self, positions: np.ndarray, diameters: np.ndarray, edges: np.ndarray) -> None:
        """
        positions : (N, 3) vertex coordinates; diameters : (N,);
        edges : (E, 2) vertex index pairs, one per drawn segment
        """
        self.edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        self.p0 = positions[self.edges[:, 0]]
        self.p1 = positions[self.edges[:, 1]]
        self.r0 = diameters[self.edges[:, 0]] / 2.0
        self.r1 = diameters[self.edges[:, 1]] / 2.0
        self.u = self.p1 - self.p0
        self.uu = np.einsum("ij,ij->i", self.u, self.u)

    def pick(
        self, origin: np.ndarray, direction: np.ndarray, pixel_size: float, tolerance: float = 4.0
    ) -> Union[tuple, None]:
        """
        origin, direction : the ray (direction need not be normalized).
        pixel_size : size of one screen pixel in world units at unit distance
        from the camera; tolerance : extra pick radius in pixels.
        Returns (edge index, t, hit point, depth) or None; t (0..1) is the
        position of the closest approach along the edge, from its first vertex,
        and depth the distance along the ray to where it meets the surface.
        Segments the ray actually passes through come first (the surface met
        first wins); the pixel tolerance is used only when there are none, so
        that it can never select something behind a visible surface.
        """
        if len(self.edges) == 0:
            return None
        d = direction / np.linalg.norm(direction)
        w = origin - self.p0
        b = self.u @ d
        e = np.einsum("ij,ij->i", self.u, w)
        dw = w @ d
        denom = self.uu - b * b  # |d| = 1
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(denom > 1e-12, (e - b * dw) / denom, 0.0)
            t = np.where(self.uu > 0, t, 0.0)
        t = np.clip(t, 0.0, 1.0)
        closest = self.p0 + t[:, None] * self.u
        s = (closest - origin) @ d  # distance along the ray
        miss = np.linalg.norm(origin + s[:, None] * d - closest, axis=1)
        radius = self.r0 + t * (self.r1 - self.r0)
        hit = (s > 0) & (miss <= radius)
        if not np.any(hit):  # nothing under the mouse itself: allow a few pixels
            radius = radius + tolerance * pixel_size * np.maximum(s, 0.0)
            hit = (s > 0) & (miss <= radius)
            if not np.any(hit):
                return None
        candidates = np.flatnonzero(hit)
        entry = self._entry(origin, d, candidates, radius[candidates])
        k = int(candidates[np.argmin(entry)])
        point = origin + entry.min() * d  # on the surface, where the ray meets it
        return k, float(t[k]), point, float(entry.min())

    def _entry(self, origin: np.ndarray, d: np.ndarray, k: np.ndarray, radius: np.ndarray) -> np.ndarray:
        """
        distance along the ray to where it enters each capsule *k* (a cylinder
        of the given radius with rounded ends): through the side, at the angle
        between the ray and the segment, or else through an end cap
        """
        p0, u, uu = self.p0[k], self.u[k], self.uu[k]
        length = np.sqrt(uu)
        cos = np.where(length > 0, (u @ d) / np.maximum(length, 1e-12), 1.0)
        sin = np.sqrt(np.maximum(1.0 - cos * cos, 0.0))
        # closest approach of the ray to the (unbounded) axis line
        w = origin - p0
        b = u @ d
        denom = uu - b * b
        with np.errstate(divide="ignore", invalid="ignore"):
            t_line = np.where(denom > 1e-12, (np.einsum("ij,ij->i", u, w) - b * (w @ d)) / denom, 0.5)
        on_axis = p0 + t_line[:, None] * u
        s_line = (on_axis - origin) @ d
        miss = np.linalg.norm(origin + s_line[:, None] * d - on_axis, axis=1)
        half_chord = np.sqrt(np.maximum(radius * radius - miss * miss, 0.0)) / np.maximum(sin, 1e-6)
        side = s_line - half_chord
        t_side = t_line - half_chord * b / np.maximum(uu, 1e-12)
        side_ok = (miss <= radius) & (sin > 1e-6) & (t_side >= 0.0) & (t_side <= 1.0) & (side > 0)
        best = np.where(side_ok, side, np.inf)
        for end in (p0, p0 + u):  # rounded end caps
            s_end = (end - origin) @ d
            miss_end = np.linalg.norm(origin + s_end[:, None] * d - end, axis=1)
            cap = s_end - np.sqrt(np.maximum(radius * radius - miss_end * miss_end, 0.0))
            best = np.where((miss_end <= radius) & (cap > 0) & (cap < best), cap, best)
        # a ray that grazes the capsule without meeting a side or cap (rounding): closest approach
        fallback = (np.clip(t_line, 0, 1)[:, None] * u + p0 - origin) @ d
        return np.where(np.isfinite(best), best, fallback)

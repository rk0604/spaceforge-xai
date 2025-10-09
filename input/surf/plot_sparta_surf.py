#!/usr/bin/env python3

"""
plot_sparta_surf.py — visualize SPARTA .surf (Format B) triangle meshes in 3D.

Usage:
  python plot_sparta_surf.py path/to/cupola.surf
  python plot_sparta_surf.py path/to/cupola.surf --save cupola.png
  python plot_sparta_surf.py path/to/cupola.surf --show-normals

Notes:
- Expects the SPARTA "Format B" used in your files:
    <header line>
    "<N> triangles"
    "Triangles"
    then N lines with: id x1 y1 z1 x2 y2 z2 x3 y3 z3
- Single-figure render (one mesh per run). If you pass multiple files,
  they will be drawn in the same figure as separate Poly3DCollections.

"""
import argparse
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def parse_surf(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a SPARTA .surf (Format B) file.

    Returns:
        vertices: (M, 3) array of all vertices (deduplicated)
        faces:    (T, 3) int array of vertex indices into vertices
    """
    lines = file_path.read_text().strip().splitlines()
    # find the "Triangles" line
    try:
        tri_idx = next(i for i, ln in enumerate(lines) if ln.strip().lower().startswith("triangles"))
    except StopIteration:
        raise ValueError(f"No 'Triangles' section found in {file_path}")

    raw = []
    for ln in lines[tri_idx + 1:]:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        # Expect: id + 9 numbers = 10 tokens
        if len(parts) != 10:
            # Allow comments or accidental extra lines after triangles
            # Stop if we hit a non-data line.
            try:
                _ = int(parts[0])
                # If first token is int but wrong length, it's malformed
                raise ValueError(f"Malformed triangle line in {file_path}: {ln}")
            except ValueError:
                break
        tri_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        raw.append((tri_id, coords))

    # Build vertices and faces, deduplicating vertices using a tolerance
    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    tol = 1e-12

    def find_or_add(v: Tuple[float, float, float]) -> int:
        # Simple linear search with tolerance — fine for small meshes
        for i, w in enumerate(verts):
            if abs(v[0] - w[0]) < tol and abs(v[1] - w[1]) < tol and abs(v[2] - w[2]) < tol:
                return i
        verts.append(v)
        return len(verts) - 1

    for _, coords in raw:
        v1 = (coords[0], coords[1], coords[2])
        v2 = (coords[3], coords[4], coords[5])
        v3 = (coords[6], coords[7], coords[8])
        i1, i2, i3 = find_or_add(v1), find_or_add(v2), find_or_add(v3)
        faces.append((i1, i2, i3))

    return np.array(verts, dtype=float), np.array(faces, dtype=int)


def compute_face_centers_and_normals(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-face centers and (unnormalized) normals.
    """
    p1 = vertices[faces[:, 0]]
    p2 = vertices[faces[:, 1]]
    p3 = vertices[faces[:, 2]]
    centers = (p1 + p2 + p3) / 3.0
    normals = np.cross(p2 - p1, p3 - p1)
    return centers, normals


def set_axes_equal(ax):
    """
    Make 3D axes have equal scale so that spheres look like spheres, etc.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)
    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])


def plot_meshes(meshes, show_normals=False, save_path: Path = None, title: str = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    poly_sets = []
    for (vertices, faces), label in meshes:
        tris = [vertices[f] for f in faces]
        poly = Poly3DCollection(tris, linewidths=0.5, alpha=0.7)
        poly.set_edgecolor("k")  # do not set face color explicitly
        ax.add_collection3d(poly)
        poly_sets.append((vertices, faces, label))

        if show_normals:
            centers, normals = compute_face_centers_and_normals(vertices, faces)
            # Normalize for visualization and scale arrows
            lengths = np.linalg.norm(normals, axis=1) + 1e-12
            normals_dir = normals / lengths[:, None]
            scale = 0.05 * np.max(np.ptp(vertices, axis=0))
            ax.quiver(
                centers[:, 0], centers[:, 1], centers[:, 2],
                normals_dir[:, 0], normals_dir[:, 1], normals_dir[:, 2],
                length=scale, normalize=False
            )

    # Autoscale based on all vertices
    all_pts = np.vstack([v for (v, f, l) in poly_sets for v in [v]])
    ax.auto_scale_xyz(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2])
    set_axes_equal(ax)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    if title:
        ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize SPARTA .surf (Format B) triangle meshes in 3D.")
    parser.add_argument("files", nargs="+", type=Path, help="One or more .surf files")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save a PNG")
    parser.add_argument("--show-normals", action="store_true", help="Draw per-face normals (for orientation checks)")
    parser.add_argument("--title", type=str, default=None, help="Optional plot title")
    args = parser.parse_args()

    meshes = []
    for fp in args.files:
        verts, faces = parse_surf(fp)
        meshes.append(((verts, faces), fp.name))

    title = args.title if args.title is not None else ", ".join([lbl for (_, lbl) in meshes])
    plot_meshes(meshes, show_normals=args.show_normals, save_path=args.save, title=title)


if __name__ == "__main__":
    main()

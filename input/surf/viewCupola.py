#!/usr/bin/env python3
"""
view_cupola_surf.py — Minimal SPARTA .surf Triangles viewer (format B).
Usage:
  python view_cupola_surf.py /path/to/cupola.surf [--save]
If --save is given, saves PNGs for a few preset angles next to the .surf file.
"""
import sys, os, math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def read_surf_triangles(path):
    tris = []
    with open(path, "r") as f:
        lines = f.readlines()
    i = 0
    # find "Triangles" section
    while i < len(lines) and lines[i].strip() != "Triangles":
        i += 1
    if i >= len(lines):
        raise RuntimeError("Triangles section not found in surf file.")
    i += 1  # skip the next line per SPARTA spec
    if i < len(lines): i += 1
    # read remaining triangle lines until EOF or blank
    for j in range(i, len(lines)):
        line = lines[j].strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # Expect: id + 9 floats (format B). Ignore any extras.
        if len(parts) < 10:
            continue
        # id = int(parts[0])  # unused
        vals = list(map(float, parts[1:10]))
        tri = np.array(vals, dtype=float).reshape(3,3)
        tris.append(tri)
    if not tris:
        raise RuntimeError("No triangles parsed from Triangles section.")
    return np.array(tris, dtype=float)

def plot_tris(ax, tri_array, alpha=1.0):
    pc = Poly3DCollection(tri_array, linewidths=0.2, edgecolors='k', alpha=alpha)
    ax.add_collection3d(pc)

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    path = sys.argv[1]
    do_save = ("--save" in sys.argv[2:])
    tris = read_surf_triangles(path)
    allv = tris.reshape(-1,3)
    mins = allv.min(axis=0); maxs = allv.max(axis=0)
    size = (maxs - mins).max()
    center = (maxs + mins) / 2.0

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    plot_tris(ax, tris, alpha=0.85)
    ax.set_box_aspect((1,1,1))
    ax.set_xlim(center[0]-size/2, center[0]+size/2)
    ax.set_ylim(center[1]-size/2, center[1]+size/2)
    ax.set_zlim(center[2]-size/2, center[2]+size/2)
    ax.set_title(os.path.basename(path))
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    plt.tight_layout()
    if do_save:
        base = os.path.splitext(path)[0]
        for name, elev, az in [("iso",30,45),("front",0,0),("top",90,0),("side",0,90)]:
            ax.view_init(elev=elev, azim=az)
            out = f"{base}_{name}.png"
            plt.savefig(out, dpi=160)
        print("Saved views next to the .surf file.")
    plt.show()

if __name__ == "__main__":
    main()

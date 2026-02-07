#!/usr/bin/env python3
"""
visualizerScene.py

Usage (your setup)
- Run from: spaceforge-xai/input
- Expects:
    in.wake_harness
    surf/*.surf
- Writes images to:
    surf/vis/

Run:
  cd spaceforge-xai/input
  python3 visualizerScene.py

Outputs:
  surf/vis/scene_iso.png
  surf/vis/scene_x_plus.png
  surf/vis/scene_x_minus.png
  surf/vis/scene_y_plus.png
  surf/vis/scene_y_minus.png
  surf/vis/scene_z_plus.png
  surf/vis/scene_z_minus.png
  surf/vis/scene_iso2.png
"""

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ----------------------------
# USER SETTINGS
# ----------------------------
DECK_PATH = Path("in.wake_harness")
SURF_ROOT = Path(".")                 # contains the "surf/" folder
OUT_DIR   = SURF_ROOT / "surf" / "vis"

MAX_TRIS_PER_OBJECT = None            # None = all triangles
ALPHA = 0.35
EDGE_COLOR = "k"
EDGE_WIDTH = 0.6
DPI = 300
FIGSIZE = (11, 8.5)

# Views to render: (name, elev, azim)
VIEWS = [
    ("scene_iso",    18,  -60),
    ("scene_iso2",   28,   35),
    ("scene_x_plus",  0,    0),   # looking along +x toward origin-ish
    ("scene_x_minus", 0,  180),   # looking along -x
    ("scene_y_plus",  0,   90),   # looking along +y
    ("scene_y_minus", 0,  -90),   # looking along -y
    ("scene_z_plus", 90,    0),   # looking down +z
    ("scene_z_minus",-90,   0),   # looking up from -z
]

# ----------------------------
# PARSERS
# ----------------------------
_float_re = re.compile(r"[-+]?(?:(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)")

def parse_read_surf_lines(deck_text: str):
    objs = []
    for raw in deck_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line or not line.startswith("read_surf"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        surf_path = Path(parts[1])

        dx = dy = dz = 0.0
        if "trans" in parts:
            k = parts.index("trans")
            if k + 3 >= len(parts):
                raise ValueError(f"Malformed trans in line: {raw}")
            dx = float(parts[k+1]); dy = float(parts[k+2]); dz = float(parts[k+3])

        objs.append({"path": surf_path, "trans": (dx, dy, dz)})
    return objs


def load_sparta_surf_triangles(path: Path):
    text = path.read_text(errors="ignore").splitlines()

    n_tris = None
    tri_start = None
    for i, line in enumerate(text):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"^\s*(\d+)\s+triangles\s*$", s, re.IGNORECASE)
        if m:
            n_tris = int(m.group(1))
            continue
        if s.lower() == "triangles":
            tri_start = i + 1
            break

    if n_tris is None or tri_start is None:
        raise ValueError(f"[{path}] Could not find '<N> triangles' and 'Triangles' header.")

    tris = []
    bad_lines = 0

    for j in range(tri_start, len(text)):
        line = text[j].strip()
        if not line or line.startswith("#"):
            continue

        nums = _float_re.findall(line)
        if len(nums) >= 10:
            coords = list(map(float, nums[1:10]))  # skip id
        elif len(nums) >= 9:
            coords = list(map(float, nums[:9]))
        else:
            bad_lines += 1
            continue

        tri = np.array(coords, dtype=float).reshape(3, 3)
        tris.append(tri)

        if MAX_TRIS_PER_OBJECT is not None and len(tris) >= MAX_TRIS_PER_OBJECT:
            break

    tris = np.stack(tris, axis=0) if tris else np.zeros((0, 3, 3), dtype=float)
    if tris.shape[0] == 0:
        raise ValueError(f"[{path}] Parsed 0 triangles (bad_lines={bad_lines}).")

    return tris


# ----------------------------
# PLOTTING
# ----------------------------
def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0]); x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0]); y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0]); z_mid = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid - plot_radius, x_mid + plot_radius])
    ax.set_ylim3d([y_mid - plot_radius, y_mid + plot_radius])
    ax.set_zlim3d([z_mid - plot_radius, z_mid + plot_radius])


def add_mesh(ax, tris, label=None):
    poly = Poly3DCollection(
        tris,
        alpha=ALPHA,
        linewidths=EDGE_WIDTH,
        edgecolors=EDGE_COLOR
    )
    ax.add_collection3d(poly)

    if label is not None:
        pts = tris.reshape(-1, 3)
        c = pts.mean(axis=0)
        ax.text(c[0], c[1], c[2], label, fontsize=10)


def plot_and_save_views(loaded, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pts = np.vstack([it["tris"].reshape(-1, 3) for it in loaded])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)

    for name, elev, azim in VIEWS:
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(111, projection="3d")

        for it in loaded:
            add_mesh(ax, it["tris"], label=it["name"])

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])
        set_axes_equal(ax)

        ax.view_init(elev=elev, azim=azim)

        plt.tight_layout()
        out_path = out_dir / f"{name}.png"
        plt.savefig(out_path, dpi=DPI)
        plt.close(fig)
        print(f"Saved: {out_path}")


def main():
    if not DECK_PATH.is_file():
        raise FileNotFoundError(f"Deck not found: {DECK_PATH.resolve()}")

    deck_text = DECK_PATH.read_text(errors="ignore")
    objs = parse_read_surf_lines(deck_text)
    if not objs:
        raise RuntimeError(f"No read_surf lines found in {DECK_PATH}")

    loaded = []
    for obj in objs:
        surf_path = (SURF_ROOT / obj["path"]).resolve()
        dx, dy, dz = obj["trans"]

        if not surf_path.is_file():
            raise FileNotFoundError(
                f"Could not find surf file: {surf_path}\n"
                f"Deck token was: {obj['path']}\n"
                f"Run from spaceforge-xai/input (the folder containing surf/)."
            )

        tris = load_sparta_surf_triangles(surf_path)
        tris = tris + np.array([dx, dy, dz], dtype=float)[None, None, :]

        loaded.append({
            "name": obj["path"].name,
            "tris": tris,
            "trans": (dx, dy, dz),
            "path": str(obj["path"]),
        })

    print("\n=== Objects loaded ===")
    for it in loaded:
        print(f"- {it['name']}  trans={it['trans']}  deck={it['path']}  tris={it['tris'].shape[0]}")

    plot_and_save_views(loaded, OUT_DIR)


if __name__ == "__main__":
    main()

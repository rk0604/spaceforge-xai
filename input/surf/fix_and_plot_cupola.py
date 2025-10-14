import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def read_surf(path):
    verts = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 10 and parts[0].isdigit():
                _, *nums = parts
                nums = list(map(float, nums))
                tri = np.array(nums).reshape(3, 3)
                verts.append(tri)
    return np.array(verts)

tris = read_surf("cupola.surf")

# --- Basic sanity checks ---
print(f"{len(tris)} triangles read")
areas = 0.5*np.linalg.norm(np.cross(tris[:,1]-tris[:,0], tris[:,2]-tris[:,0]),axis=1)
print(f"Min area: {areas.min():.4e},  Zero-area facets: {(areas<1e-10).sum()}")

# remove degenerate / duplicate triangles
mask = areas > 1e-10
tris = tris[mask]

# optional: flip triangles to make normals consistent along +x
tris[...,0] *= -1

# write cleaned file
with open("cupola_clean.surf","w") as f:
    f.write("# cleaned SPARTA surface\n%d triangles\nTriangles\n\n"%len(tris))
    for i,t in enumerate(tris,1):
        f.write(f"{i} " + " ".join(f"{x:.6e}" for x in t.flatten()) + "\n")

# visualize
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.add_collection3d(Poly3DCollection(tris,alpha=0.4,edgecolor='k'))
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
plt.show()

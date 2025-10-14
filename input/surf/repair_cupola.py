import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

tris = []
with open("cupola_clean.surf") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 10 and parts[0].isdigit():
            _, *vals = parts
            pts = np.array(vals, float).reshape(3,3)
            tris.append(pts)
tris = np.array(tris)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.add_collection3d(Poly3DCollection(tris, facecolor='lightblue', edgecolor='k', alpha=0.6))
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
plt.show()

import numpy as np
import random
from vedo import *

surf = Mesh("conical_nozzle.stl")

# --- Create binarized volume ---
vol = surf.binarize(spacing=(0.001, 0.001, 0.005))

# Duplicate the volume for before/after comparison
vol_before = vol.clone()
vol_after  = vol.clone()

# Random shear stress values
pts = vol_after.points
random.seed(42)
shear_vals = [random.uniform(0, 90) for _ in range(len(pts))]

# Apply colormap ONLY to vol_after
vol_after.pointdata["Shear Stress"] = shear_vals
vol_after.cmap("plasma", shear_vals, vmax=np.max(shear_vals), vmin=np.min(shear_vals))
vol_after.alpha(1)
vol_after.add_scalarbar(title="Shear Stress (Pa)", c="w")

# --- Create Plotter with 3 panels ---
plt = Plotter(N=3, axes=1)

# Panel 0: original STL mesh
plt.at(0).show(surf, "Original STL Mesh")

# Panel 1: binarized volume BEFORE cmap
plt.at(1).show(vol_before, "Binarized Volume (Before cmap)")

# Panel 2: binarized volume AFTER cmap
plt.at(2).show(vol_after, "Binarized Volume (After cmap)")

plt.interactive().close()

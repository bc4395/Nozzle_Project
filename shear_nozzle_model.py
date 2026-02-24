import numpy as np
from vedo import *
import pandas as pd
from scipy.optimize import curve_fit

#=================================================================================================
# ----------------- Power-law Viscosity Model -----------------
def model(sr, K, n):
    return K * np.power(sr, n - 1)

# ----------------- Flow Rate -----------------
def calculate_flow_rate(r1, r2, L, K, n, delta_P):
    z_vals = np.linspace(0.001, L, 100)
    dz = z_vals[1] - z_vals[0]
    flow = 0.0

    for z in z_vals:
        Rz = r1 - (r1 - r2) * z / L
        dP_dz = delta_P / L
        vz_max = ((dP_dz * Rz) / (2 * K))**(1/n)
        Qz = (np.pi * Rz**2 * vz_max) / (3*n + 1) * (n + 1)
        flow += Qz * dz
    return flow

# ----------------- Compute shear stress -----------------
def compute_shear_stress(x, y, z, R_in, R_out, L, Q, K, n,):
    r_vals = np.sqrt(x**2 + y**2)
    Rz_vals = R_in - (R_in - R_out) * ((L - z) / L)
    shear_vals = np.zeros_like(r_vals)

    nonzero = r_vals > 0

    gamma_dot = (((n + 1) / n) * (2 * Q / (np.pi * Rz_vals[nonzero]**3))
                 * (r_vals[nonzero] / Rz_vals[nonzero]) ** (1 / n))

    shear_vals[nonzero] = K * np.abs(gamma_dot) ** n
    return shear_vals

# ----------------- Nozzzle Geometry Setup -----------------
R_in = 0.00175     # Base at z = L (m)
R_out = 0.0004318  # Tip at z = 0 (m)
L = 0.0314         # Length (m)

df = pd.read_csv("A4C4.csv")
sr_data = df["SR"].values
vis_data = df["Vis"].values
K, n = curve_fit(model, sr_data, vis_data, p0=[1.0, 1.0])[0]

# ----------------- Pressure Input & Flow -----------------
pressure_psi = float(input("Enter pressure used (psi): "))
pressure_pa = pressure_psi * 6894.76
Q = calculate_flow_rate(R_in, R_out, L, K, n, pressure_pa)

#=================================================================================================



#=================================================================================================
#----------------- Nozzle Mesh Analysis -----------------

nozzle_mesh = Mesh("conical_nozzle.stl")
nozzle_pts = nozzle_mesh.binarize(spacing=(0.00005, 0.00005, 0.00005))

n_pts = nozzle_pts.topoints()

n_coords = n_pts.points
n_colors = n_pts.pointcolors

# Remove all bordered points
black = np.all(n_colors[:, :3] == [0, 0, 0], axis=1)
mask = ~black

# Create new Points object without border points
nozzle_cpts = Points(n_coords[mask])
nozzle_cpts.ps(n_pts.ps())  # copy point size
nozzle_cpts.pointcolors = n_colors[mask]

# Retrieve coordinates in x, y, z arrays
x_nvals = nozzle_cpts.coordinates[:, 0]
y_nvals = nozzle_cpts.coordinates[:, 1]
z_nvals = nozzle_cpts.coordinates[:, 2]

# Compute shear stress for each point
nozzle_shear_vals = compute_shear_stress(
    x_nvals, y_nvals, z_nvals,
    R_in, R_out, L, Q, K, n)

# Apply colormap ONLY to vol_after
nozzle_cpts.pointdata["Shear Stress"] = nozzle_shear_vals
nozzle_cpts.cmap("plasma", nozzle_shear_vals, on="points", vmax=np.max(nozzle_shear_vals), vmin=0)
nozzle_cpts.add_scalarbar(title="Shear Stress (Pa)", c="w")

Plotter().show(nozzle_cpts, axes=1, bg="black", title=f"NOZZLE STRESS DISTRIBUTION at P={pressure_pa}Pa")

#=================================================================================================



#=================================================================================================
#----------------- Cell Mesh Analysis -------------------
# Center position of cell to insert
x_insert = 0.000005
y_insert = 0.000005
z_insert = 0.00

# cell_mesh = Mesh("random.stl")
# cell_mesh = Sphere(pos=(x_insert,y_insert,z_insert), r=0.00001)
cell_mesh = Ellipsoid(pos=(x_insert, y_insert, z_insert), axis1=0.00001, axis2=0.000005, axis3=0.000005)

cell_pts = cell_mesh.binarize(spacing=(0.0000005, 0.0000005, 0.0000005))

c_pts = cell_pts.topoints()

c_coords = c_pts.points
c_colors = c_pts.pointcolors

# Remove all bordered points
black = np.all(c_colors[:, :3] == [0, 0, 0], axis=1)
mask = ~black

# Create new Points object without border points
cell_cpts = Points(c_coords[mask])
cell_cpts.ps(c_pts.ps())  # copy point size
cell_cpts.pointcolors = c_colors[mask]

# Retrieve coordinates in x, y, z arrays
x_cvals = cell_cpts.coordinates[:, 0]
y_cvals = cell_cpts.coordinates[:, 1]
z_cvals = cell_cpts.coordinates[:, 2]

# Compute shear stress for each point
cell_shear_vals = compute_shear_stress(
    x_cvals, y_cvals, z_cvals,
    R_in, R_out, L, Q, K, n)

cell_cpts.pointdata["Shear Stress"] = cell_shear_vals
cell_cpts.cmap("plasma", cell_shear_vals, on="points", vmax=np.max(cell_shear_vals), vmin=0)
cell_cpts.add_scalarbar(title="Shear Stress (Pa)", c="w")

Plotter().show(cell_cpts, axes=1, bg="black", title=f"CELL STRESS DISTRIBUTION at P={pressure_pa}Pa")

#=================================================================================================



#=================================================================================================
#----------------- Cross-Sectional Analysis -------------------
z_level_mm = input("Enter z-level for cross-sectional analysis (0mm - 31.4mm): ")
z_level_m = float(z_level_mm) / 1000.0

# Calculate corresponding radius at given z-level
radius_at_z = R_in - (R_in - R_out) * ((L - z_level_m) / L)

cross_section = Disc(pos=(0, 0, 0), r1=0, r2=radius_at_z)
cross_x_vals = cross_section.coordinates[:, 0]
cross_y_vals = cross_section.coordinates[:, 1]
cross_z_vals = np.full(cross_x_vals.shape, z_level_m)

# Compute shear stress for each point
cross_shear_vals = compute_shear_stress(
    cross_x_vals, cross_y_vals, cross_z_vals,
    R_in, R_out, L, Q, K, n)

cross_section.pointdata["Shear Stress"] = cross_shear_vals
cross_section.cmap("plasma", cross_shear_vals, on="points", vmax=np.max(nozzle_shear_vals), vmin=0)
cross_section.add_scalarbar(title="Shear Stress (Pa)", c="w")

Plotter().show(cross_section, axes=1, bg="black", title=f"CROSS-SECTION DISTRIBUTION at z={z_level_mm}mm")

#=================================================================================================
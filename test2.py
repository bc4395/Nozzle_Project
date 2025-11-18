import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from vedo import *

# ----------------- Power-law Viscosity Model -----------------
def model(sr, K, n):
    return K * np.power(sr, n - 1)

# ----------------- Shear Stress Calculation -----------------
def compute_shear_stress(r, Rz, Q, K, n):
    if r == 0:
        return 0.0
    gamma_dot = ((n + 1) / n) * (2 * Q / (np.pi * Rz**3)) * (r / Rz)**(1 / n)
    return K * np.abs(gamma_dot)**n

# ----------------- Flow Rate via Numerical Integration -----------------
def calculate_flow_rate(r1, r2, L, K, n, delta_P):
    z_vals = np.linspace(0.001, L, 100)
    dz = z_vals[1] - z_vals[0]
    flow_sum = 0
    for z in z_vals:
        Rz = r1 - (r1 - r2) * z / L
        dP_dz = delta_P / L
        vz_max = ((dP_dz * Rz) / (2 * K))**(1/n)
        Qz = (np.pi * Rz**2 * vz_max) / (3*n + 1) * (n + 1)
        flow_sum += Qz * dz
    return flow_sum

# ----------------- Geometry & Grid Setup -----------------
R_in = 0.00175     # Base at z = L (m)
R_out = 0.0004318  # Tip at z = 0 (m)
L = 0.0314         # Length (m)

nz = 101
nr = 50
ntheta = 360

z_vals = np.linspace(L, 0, nz)
y_vals = np.linspace(-R_in, R_in, 2*nr)
x_vals = np.linspace(-R_in, R_in, 2*nr)

X, Y = np.meshgrid(x_vals, y_vals)

# ----------------- Fit Viscosity Data -----------------
df = pd.read_csv("A4C4.csv")
sr_data = df["SR"].values
vis_data = df["Vis"].values
K, n = curve_fit(model, sr_data, vis_data, p0=[1.0, 1.0])[0]

# ----------------- Pressure Input & Flow -----------------
pressure_psi = float(input("Enter pressure used (psi): "))
pressure_pa = pressure_psi * 6894.76
Q = calculate_flow_rate(R_in, R_out, L, K, n, pressure_pa)

# ----------------- Compute 3D Shear Volume -----------------
shear_volume = np.zeros((nz, len(y_vals), len(x_vals)))
cloud_pts = {}
for i, z in enumerate(z_vals):
    Rz = R_in - (R_in - R_out) * ((L - z) / L)
    for j, y in enumerate(y_vals):
        for k, x in enumerate(x_vals):
            r = np.sqrt(x**2 + y**2)
            shear_volume[i, j, k] = compute_shear_stress(r, Rz, Q, K, n)
            cloud_pts[(z, y, x)] = shear_volume[i, j, k]

# ----------------- Interpolator -----------------
flipped_z_vals = z_vals[::-1]
interpolator = RegularGridInterpolator(
    (flipped_z_vals, y_vals, x_vals),
    shear_volume[::-1],
    bounds_error=False,
    method='linear',
    fill_value=None
)

# ----------------- Load STL Nozzle -----------------
mesh = Mesh("conical_nozzle.stl")
pts = mesh.points
coords = np.c_[pts[:, 2], pts[:, 1], pts[:, 0]]  # (z, y, x)
shear_vals = interpolator(coords)

cmap = "plasma"

mesh.pointdata["Shear Stress"] = shear_vals
mesh.cmap(cmap, shear_vals, on="points", vmax=np.max(shear_vals), vmin=np.min(shear_vals))
mesh.alpha(1)
mesh.add_scalarbar(title="Shear Stress (Pa)", c="w")
mesh.write("shear_nozzle.ply")
mesh.write("shear_nozzle.obj")

# ----------------- First Plot: Just Nozzle -----------------
plt1 = Plotter(title="3D Shear Field", size=(900, 700), axes=1, bg="k")
plt1.show(mesh, zoom=1.2, viewup="z", interactive=True)
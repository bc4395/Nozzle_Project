import numpy as np

# Function to compute shear stress at each point based on radial position and flow
def compute_shear_stress(x, y, z, R_in, R_out, L, P, K, n):
    Q = (np.pi * R_in**3 / (3*n + 1)) * ((R_in * P) / (2 * K * L))**(1/n)

    r_vals = np.sqrt(x**2 + y**2)  # Radial distance from center
    Rz = ((R_in - R_out) / L) * (z - L) + R_in  # Radial position along nozzle length

    # ---- FLOW-RATE-BASED PRESSURE GRADIENT ----
    term = ((3*n + 1) * Q) / (np.pi * Rz**3)
    dP_dz = (2 * K / Rz) * (term ** n)

    # ---- SHEAR RATE (consistent with power-law physics) ----
    gamma_dot = ((r_vals / 2) * abs(dP_dz) / K) ** (1/n)

    shear_vals = K * np.abs(gamma_dot) ** n  # Shear stress based on the power law
    return shear_vals
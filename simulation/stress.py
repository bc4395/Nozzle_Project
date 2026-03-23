import numpy as np

# Function to compute shear stress at each point based on radial position and flow
def compute_shear_stress(x, y, z, R_in, R_out, L, Q, K, n):
    r_vals = np.sqrt(x**2 + y**2)  # Radial distance from center
    Rz_vals = R_in - (R_in - R_out) * ((L - z) / L)  # Radial position along nozzle length
    shear_vals = np.zeros_like(r_vals)

    nonzero = r_vals > 0  # Avoid division by zero

    # Shear rate (gamma_dot) at each point
    gamma_dot = (((n + 1) / n) * (2 * Q / (np.pi * Rz_vals[nonzero]**3))
                 * (r_vals[nonzero] / Rz_vals[nonzero]) ** ((1 / n)-1))

    shear_vals[nonzero] = K * np.abs(gamma_dot) ** n  # Shear stress based on the power law
    return shear_vals
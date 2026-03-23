import numpy as np

# Function to calculate flow rate using nozzle geometry and pressure
def calculate_flow_rate(r1, r2, L, K, n, delta_P):
    z_vals = np.linspace(0.001, L, 100)
    dz = z_vals[1] - z_vals[0]
    flow = 0.0

    for z in z_vals:
        Rz = r1 - (r1 - r2) * z / L  # Radial position at z
        dP_dz = delta_P / L  # Pressure gradient along the length
        vz_max = ((dP_dz * Rz) / (2 * K))**(1/n)  # Max velocity at radial position
        Qz = (np.pi * Rz**2 * vz_max) / (3*n + 1) * (n + 1)  # Flow rate at z
        flow += Qz * dz  # Accumulate flow over the length

    return flow
import numpy as np
from scipy.optimize import curve_fit

# Power-law model function
def model(sr, K, n):
    return K * np.power(sr, n - 1)

# Function to fit power-law model to SR (shear rate) and viscosity data
def fit_power_law(df):
    sr_data = df["SR"].values
    vis_data = df["Vis"].values
    K, n = curve_fit(model, sr_data, vis_data, p0=[1.0, 1.0])[0]  # Initial guess for K, n
    K = K / 1000  # Convert K to Pa.s^n from mPa.s^n
    return K, n
import os
import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from vedo import *
from scipy.optimize import curve_fit
from simulation.rheology import model
from simulation.stress import compute_shear_stress


class CellWindow(QWidget):

    def __init__(self, parent):
        super().__init__()

        self.parent = parent
        self.setWindowTitle("Cell Simulation")
        self.setFixedWidth(300)

        layout = QVBoxLayout()

        zlabel = QLabel("Enter z-level (0mm - 31.4mm):")
        layout.addWidget(zlabel)

        self.cell_z = QLineEdit()
        layout.addWidget(self.cell_z)

        cell_z_button = QPushButton("Enter Z-level")
        cell_z_button.clicked.connect(self.xycoords)
        layout.addWidget(cell_z_button)
        
        # X/Y container
        self.xy_widget = QWidget()
        xy_layout = QVBoxLayout()

        self.xlabel = QLabel("Enter x-level:")
        xy_layout.addWidget(self.xlabel)

        self.cell_x = QLineEdit()
        xy_layout.addWidget(self.cell_x)
    
        self.ylabel = QLabel("Enter y-level:")
        xy_layout.addWidget(self.ylabel)

        self.cell_y = QLineEdit()
        xy_layout.addWidget(self.cell_y)

        xy_button = QPushButton("Run Cell Simulation")
        xy_button.clicked.connect(self.run_cell)
        xy_layout.addWidget(xy_button)

        self.xy_widget.setLayout(xy_layout)

        # Hide X/Y inputs at start
        self.xy_widget.hide()

        layout.addWidget(self.xy_widget)

        self.setLayout(layout)

    
    # ---------------- STEP 1: HANDLE Z INPUT ----------------
    def xycoords(self):

        try:
            z_mm = float(self.cell_z.text())
        except:
            QMessageBox.warning(self, "Error", "Invalid z value")
            return

        z_m = z_mm / 1000

        R_in = self.parent.R_in
        R_out = self.parent.R_out
        L = self.parent.L

        # Compute radius at this z
        self.radius = R_in - (R_in - R_out) * ((L - z_m) / L)

        radius_mm = self.radius * 1000

        # Update labels dynamically
        self.xlabel.setText(f"Enter x-level (±{radius_mm:.4f} mm):")
        self.ylabel.setText(f"Enter y-level (±{radius_mm:.4f} mm):")

        # Show X/Y inputs
        self.xy_widget.show()
        self.adjustSize()


    def run_cell(self):

        R_in = self.parent.R_in
        R_out = self.parent.R_out
        L = self.parent.L
        P = self.parent.pressure_pa
        K = self.parent.K
        n = self.parent.n

        cell_mesh = Ellipsoid(pos=(float(self.cell_x.text())/1000, float(self.cell_y.text())/1000,
                                   float(self.cell_z.text())/1000), axis1=(0.00001,0,0), axis2=(0,0.000005,0),
                                   axis3=(0,0,0.000005))

        cell_pts = cell_mesh.binarize(spacing=(0.0000005, 0.0000005, 0.0000005))

        c_pts = cell_pts.topoints()

        c_coords = c_pts.points
        c_colors = c_pts.pointcolors

        # Remove all bordered points
        black = np.all(c_colors[:, :3] == [0, 0, 0], axis=1)
        mask = ~black

        # Create new Points object without border points
        cell_cpts = Points(c_coords[mask])
        cell_cpts.ps(c_pts.ps())
        cell_cpts.pointcolors = c_colors[mask]

        # Retrieve coordinates in x, y, z arrays
        x_cvals = cell_cpts.coordinates[:, 0]
        y_cvals = cell_cpts.coordinates[:, 1]
        z_cvals = cell_cpts.coordinates[:, 2]

        # Compute shear stress for each point
        cell_shear_vals = compute_shear_stress(
            x_cvals, y_cvals, z_cvals,
            R_in, R_out, L, P, K, n)

        cell_cpts.pointdata["Shear Stress"] = cell_shear_vals
        cell_cpts.cmap("plasma", cell_shear_vals, on="points", vmax=np.max(cell_shear_vals), vmin=np.min(cell_shear_vals))
        cell_cpts.add_scalarbar(title="Shear Stress (Pa)", c="w")

        plt = Plotter(title=f"{self.parent.file_name}: Cell Stress at P={self.parent.pressure_pa}Pa")
        plt.show(cell_cpts, axes=1, bg="black")



class CrossSectionWindow(QWidget):

    def __init__(self, parent):
        super().__init__()

        self.parent = parent
        self.setWindowTitle("Cross Section Simulation")
        self.setFixedWidth(300)

        layout = QVBoxLayout()

        label = QLabel("Enter z-level (0mm - 31.4mm):")
        layout.addWidget(label)

        self.z_input = QLineEdit()
        layout.addWidget(self.z_input)

        run_button = QPushButton("Run Cross Section")
        run_button.clicked.connect(self.run_cross_section)
        layout.addWidget(run_button)

        self.setLayout(layout)


    def run_cross_section(self):

        try:
            z_level_mm = float(self.z_input.text())
        except:
            QMessageBox.warning(self, "Error", "Invalid z-level")
            return

        z_level_m = z_level_mm / 1000

        R_in = self.parent.R_in
        R_out = self.parent.R_out
        L = self.parent.L
        P = self.parent.pressure_pa
        K = self.parent.K
        n = self.parent.n
        shear_max = self.parent.shear_max

        radius = R_in - (R_in - R_out) * ((L - z_level_m) / L)
        
        cross_section = Disc(r1=0, r2=radius,res=(100, 100))

        border = Plane(pos=(0, 0, z_level_m), normal=(0, 0, 1), s=(.005, .005), res=(1, 1), edge_direction=(), c="gray5", alpha=1)

        coords = cross_section.coordinates
        x = coords[:,0]
        y = coords[:,1]
        z = np.full(x.shape, z_level_m)

        shear = compute_shear_stress(x, y, z, R_in, R_out, L, P, K, n)

        cross_section.pointdata["Shear Stress"] = shear
        cross_section.cmap("plasma", shear, on="points", vmax=shear_max, vmin=0)
        cross_section.add_scalarbar(title="Shear Stress (Pa)", c="w")

        plt = Plotter(shape=(1, 2), sharecam=False, title=f"{self.parent.file_name}: Cross-Section at z={z_level_mm}mm")
        plt.show(self.parent.init_pts, border, at=0, axes=0, bg="black", viewup=[0,0,1])
        plt.show(cross_section, at=1, axes=1, bg="black", mode=10)

class SimulationApp(QWidget):

    def __init__(self):
        from PyQt5.QtWidgets import QWidget, QVBoxLayout

        super().__init__()

        self.setWindowTitle("Nozzle Flow Simulation")
        self.setFixedWidth(300)
        self.adjustSize()

        self.K = None
        self.n = None

        self.layout = QVBoxLayout()

        self.label = QLabel("Input rheology data and pressure to simulate shear stress distribution in the nozzle and on a cell.")

        self.upload_button = QPushButton("Upload Rheology CSV")
        self.upload_button.clicked.connect(self.load_csv)
        self.layout.addWidget(self.upload_button)

        # Pressure input container
        self.pressure_widget = QWidget()
        pressure_layout = QVBoxLayout()

        self.p_input_label = QLabel("Enter Pressure (psi):")
        pressure_layout.addWidget(self.p_input_label)

        self.pressure_input = QLineEdit()
        pressure_layout.addWidget(self.pressure_input)

        self.start_button = QPushButton("Start Simulation")
        self.start_button.clicked.connect(self.initialize_simulation)
        self.start_button.clicked.connect(self.initialize_nozzle)
        pressure_layout.addWidget(self.start_button)

        self.pressure_widget.setLayout(pressure_layout)

        # Hide initially
        self.pressure_widget.hide()

        # Add to main layout
        self.layout.addWidget(self.pressure_widget)

        self.nozzle_button = QPushButton("Nozzle Simulation")
        self.cell_button = QPushButton("Single Cell Simulation")
        self.cross_button = QPushButton("Cross-Section Simulation")

        self.nozzle_button.clicked.connect(self.run_nozzle)
        self.cell_button.clicked.connect(self.open_cell_window)
        self.cross_button.clicked.connect(self.open_cross_section_window)

        # Disabled until pressure is set
        self.nozzle_button.hide()
        self.cell_button.hide()
        self.cross_button.hide()

        self.layout.addWidget(self.nozzle_button)
        self.layout.addWidget(self.cell_button)
        self.layout.addWidget(self.cross_button)

        self.setLayout(self.layout)

        # Geometry Constants
        self.R_in = 0.00175 / 2
        self.R_out = 0.0004318 / 2
        self.L = 0.0314

    # Load rheology data once
    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Rheology CSV",
            "",
            "CSV Files (*.csv)"
        )

        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)
            self.file_name = os.path.basename(file_path)

            sr_data = df["SR"].values
            vis_data = df["Vis"].values

            self.K, self.n = curve_fit(model, sr_data, vis_data, p0=[1.0, 1.0])[0]

            QMessageBox.information(self, "Success", "Rheology file loaded successfully!")
            self.p_input_label.setText(f"Enter Pressure (psi) for {self.file_name}:")

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load CSV:\n{e}")

        self.pressure_widget.show()
        self.adjustSize()


    def initialize_simulation(self):
        if self.K is None:
            QMessageBox.warning(self, "Error", "Please upload a rheology CSV first.")
            return
        try:
            self.pressure_psi = float(self.pressure_input.text())
        except:
            QMessageBox.warning(self, "Error", "Invalid pressure value")
            return

        self.pressure_pa = self.pressure_psi * 6894.76

        # Enable simulation buttons
        self.nozzle_button.show()
        self.cell_button.show()
        self.cross_button.show()

        self.adjustSize()

    def initialize_nozzle(self):
        nozzle_mesh = Mesh("meshes/conical_nozzle.stl")
        nozzle_pts = nozzle_mesh.binarize(spacing=(0.000025, 0.000025, 0.000025))

        n_pts = nozzle_pts.topoints()

        n_coords = n_pts.points
        n_colors = n_pts.pointcolors

        # Remove all bordered points
        black = np.all(n_colors[:, :3] == [0, 0, 0], axis=1)
        mask = ~black

        # Create new Points object without border points
        self.init_pts = Points(n_coords[mask])
        self.init_pts.ps(n_pts.ps())
        self.init_pts.pointcolors = n_colors[mask]

        # Retrieve coordinates in x, y, z arrays
        x = self.init_pts.coordinates[:, 0]
        y = self.init_pts.coordinates[:, 1]
        z = self.init_pts.coordinates[:, 2]

        # Compute shear stress for each point
        self.shear_vals = compute_shear_stress(
            x, y, z,
            self.R_in, self.R_out, self.L, self.pressure_pa, self.K, self.n)
        
        self.shear_max = np.max(self.shear_vals)

        # Apply colormap ONLY to vol_after
        self.init_pts.pointdata["Shear Stress"] = self.shear_vals
        self.init_pts.cmap("plasma", self.shear_vals, on="points", vmax=self.shear_max, vmin=0)

    def run_nozzle(self):
        nozzle_mesh = Mesh("meshes/conical_nozzle.stl")
        nozzle_pts = nozzle_mesh.binarize(spacing=(0.000025, 0.000025, 0.000025))

        n_pts = nozzle_pts.topoints()

        n_coords = n_pts.points
        n_colors = n_pts.pointcolors

        # Remove all bordered points
        black = np.all(n_colors[:, :3] == [0, 0, 0], axis=1)
        mask = ~black

        # Create new Points object without border points
        nozzle_cpts = Points(n_coords[mask])
        nozzle_cpts.ps(n_pts.ps())
        nozzle_cpts.pointcolors = n_colors[mask]

        # Retrieve coordinates in x, y, z arrays
        x_nvals = nozzle_cpts.coordinates[:, 0]
        y_nvals = nozzle_cpts.coordinates[:, 1]
        z_nvals = nozzle_cpts.coordinates[:, 2]

        # Compute shear stress for each point
        self.nozzle_shear_vals = compute_shear_stress(
            x_nvals, y_nvals, z_nvals,
            self.R_in, self.R_out, self.L, self.pressure_pa, self.K, self.n)
        
        self.nozzle_max = np.max(self.nozzle_shear_vals)

        # Apply colormap ONLY to vol_after
        nozzle_cpts.pointdata["Shear Stress"] = self.nozzle_shear_vals
        nozzle_cpts.cmap("plasma", self.nozzle_shear_vals, on="points", vmax=self.nozzle_max, vmin=0)
        nozzle_cpts.add_scalarbar(title="Shear Stress (Pa)", c="w")

        Plotter().show(nozzle_cpts, axes=1, bg="black", title=f"{self.file_name}: Nozzle Shear Stress at P={self.pressure_pa}Pa")

    
    def open_cell_window(self):
        self.cell_window = CellWindow(self)
        self.cell_window.show()


    def open_cross_section_window(self):
        self.cross_window = CrossSectionWindow(self)
        self.cross_window.show()

# ================= Run App =================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimulationApp()
    window.show()
    sys.exit(app.exec_())
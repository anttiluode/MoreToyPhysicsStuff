# -*- coding: utf-8 -*-
import sys
import numpy as np
from scipy.ndimage import convolve
import time
from pathlib import Path
import threading
import traceback

# --- GUI and Visualization ---
# Ensure PyQt5 and pyvistaqt are installed: pip install PyQt5 pyvistaqt
try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QLabel, QSlider, QPushButton,
                                 QMessageBox, QFrame, QSizePolicy)
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
    import pyvista as pv
    from pyvistaqt import QtInteractor # Embeddable plotter
except ImportError as e:
    print("ERROR: Required libraries (PyQt5, pyvista, pyvistaqt) not found.")
    print("Please install them: pip install numpy scipy PyQt5 pyvista pyvistaqt")
    print(f"Import Error: {e}")
    sys.exit(1)

# --- Simulation Class (Physics Engine) ---
class EmergentParticleSimulator3D:
    """
    3D TADS/WoW Clockfield Simulator Engine.
    Focuses on core physics.
    """
    def __init__(self, grid_size=32):
        self.grid_size = grid_size
        print(f"Initializing 3D grid: {grid_size}x{grid_size}x{grid_size} = {grid_size**3} points")

        # --- Default Parameters ---
        self.params = {
            'dt': 0.04,
            'damping': 0.001,
            'base_c_sq': 1.0,
            'tension_factor': 5.0,
            'potential_lin': 1.0,
            'potential_cub': 0.2,
            'biharmonic_gamma': 0.02,
        }
        # Store GUI update interval here for access in worker thread
        self.update_interval_ms = 50 # Approx 20 FPS target

        # --- Internal State ---
        self.phi = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
        self.phi_old = np.zeros_like(self.phi)
        self.t = 0.0
        self.step_count = 0
        self.lock = threading.Lock() # Lock for thread-safe access to phi

        # --- 3D Laplacian Kernel ---
        self.laplacian_kernel = np.zeros((3, 3, 3), dtype=np.float64)
        self.laplacian_kernel[1, 1, 1] = -6.; self.laplacian_kernel[1, 1, 0] = 1.
        self.laplacian_kernel[1, 1, 2] = 1.; self.laplacian_kernel[1, 0, 1] = 1.
        self.laplacian_kernel[1, 2, 1] = 1.; self.laplacian_kernel[0, 1, 1] = 1.
        self.laplacian_kernel[2, 1, 1] = 1.

        self.initialize_field('gaussian_pulse_3d')

    def initialize_field(self, mode='gaussian_pulse_3d'):
        """Initialize the 3D field configuration."""
        print(f"Initializing field with mode: {mode}")
        with self.lock: # Ensure thread safety
            if mode == 'gaussian_pulse_3d':
                x, y, z = [np.arange(self.grid_size)]*3
                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                cx = cy = cz = self.grid_size // 2
                radius_sq = max((self.grid_size / 8.0)**2, 1e-6)
                self.phi[:] = 2.0 * np.exp(-((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / (2 * radius_sq))
            elif mode == 'random':
                 np.random.seed(int(time.time()))
                 self.phi = (np.random.rand(self.grid_size, self.grid_size, self.grid_size) - 0.5) * 0.1
            else: self.phi.fill(0)
            self.phi_old = np.copy(self.phi)
            self.t = 0.0
            self.step_count = 0
        print("3D Field Initialized.")

    def _laplacian(self, field): return convolve(field, self.laplacian_kernel, mode='wrap')
    def _biharmonic(self, field): return self._laplacian(self._laplacian(field))
    def _potential_deriv(self, phi): return (-self.params['potential_lin'] * phi + self.params['potential_cub'] * (phi**3))
    def _local_speed_sq(self, phi): return self.params['base_c_sq'] / (1.0 + self.params['tension_factor'] * phi**2 + 1e-9)

    def step(self):
        """Perform one time step of the 3D simulation."""
        with self.lock: # Ensure exclusive access during step calculation
            # Calculate acceleration
            lap_phi = self._laplacian(self.phi)
            biharm_phi = self._biharmonic(self.phi)
            c2 = self._local_speed_sq(self.phi)
            V_prime = self._potential_deriv(self.phi)
            acceleration = (c2 * lap_phi) - V_prime - (self.params['biharmonic_gamma'] * biharm_phi)

            # Update field using Verlet integration
            velocity = self.phi - self.phi_old
            phi_new = self.phi + (1.0 - self.params['damping']*self.params['dt'])*velocity + (self.params['dt']**2)*acceleration

            # Update state variables
            self.phi_old = self.phi
            self.phi = phi_new
            self.t += self.params['dt']
            self.step_count += 1

    def get_phi_copy(self):
        """Return a thread-safe copy of the current field."""
        with self.lock:
            return np.copy(self.phi)

    def update_params(self, new_params):
        """Update simulation parameters (thread-safe)."""
        with self.lock:
            for key, value in new_params.items():
                 if key in self.params:
                     try:
                         self.params[key] = float(value)
                     except (ValueError, TypeError):
                         print(f"Warning: Invalid value type for {key}: {value}")
                 else:
                      print(f"Warning: Attempted to update unknown parameter: {key}")

    # <<< --- ADDED MISSING METHOD --- >>>
    def reset_simulation(self):
        """Resets the simulation state by re-initializing the field."""
        # We can add options later, for now just re-initialize
        self.initialize_field('gaussian_pulse_3d') # Or 'random' or other default
    # <<< --- END ADDED METHOD --- >>>

# --- Simulation Thread ---
class SimulationWorker(QObject):
    """Runs the simulation in a separate thread."""
    finished = pyqtSignal()

    def __init__(self, simulator):
        super().__init__()
        self.simulator = simulator
        self._running = False

    def run(self):
        """Simulation loop."""
        self._running = True
        print("Simulation thread started.")
        while self._running:
            t_start = time.perf_counter()
            self.simulator.step()
            t_end = time.perf_counter()
            elapsed_ms = (t_end - t_start) * 1000
            sleep_ms = max(1, self.simulator.update_interval_ms - elapsed_ms)
            time.sleep(sleep_ms / 1000.0)

        print("Simulation thread finished.")
        self.finished.emit()

    def stop(self):
        """Signal the simulation loop to stop."""
        print("Stopping simulation thread...")
        self._running = False

# --- Main GUI Window ---
class MainWindow(QMainWindow):
    def __init__(self, simulator):
        super().__init__()
        self.simulator = simulator
        self.setWindowTitle("Interactive 3D TADS/WoW Simulation")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.plotter_frame = QFrame()
        self.plotter_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.plotter_layout = QVBoxLayout(self.plotter_frame)
        self.plotter = QtInteractor(self.plotter_frame)
        self.plotter_layout.addWidget(self.plotter)
        self.main_layout.addWidget(self.plotter_frame, stretch=3)

        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout(self.control_panel)
        self.control_panel.setFixedWidth(300)
        self.main_layout.addWidget(self.control_panel, stretch=1)

        self._setup_controls()
        self._setup_pyvista_scene()

        self.sim_thread = None
        self.sim_worker = None
        self.is_simulation_running = False

        self.gui_update_timer = QTimer(self)
        self.gui_update_timer.timeout.connect(self._update_visualization)
        self.gui_update_timer.setInterval(self.simulator.update_interval_ms)

    def _setup_controls(self):
        """Create GUI controls (buttons, sliders)."""
        control_buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self._start_simulation)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_simulation)
        self.stop_button.setEnabled(False)
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self._reset_simulation) # Connects to the corrected method below
        control_buttons_layout.addWidget(self.start_button)
        control_buttons_layout.addWidget(self.stop_button)
        control_buttons_layout.addWidget(self.reset_button)
        self.control_layout.addLayout(control_buttons_layout)

        self.control_layout.addWidget(QLabel("Parameters:"))

        self.sliders = {}
        for name, default_val in self.simulator.params.items():
            param_layout = QHBoxLayout()
            label_text = name.replace('_', ' ').title()
            label = QLabel(f"{label_text}:")
            label.setMinimumWidth(120)
            param_layout.addWidget(label)

            slider = QSlider(Qt.Horizontal)
            min_val, max_val, scale_factor, precision = self._get_slider_config(name)
            slider.setRange(int(min_val * scale_factor), int(max_val * scale_factor))
            slider.setValue(int(self.simulator.params[name] * scale_factor))
            slider.valueChanged.connect(lambda value, n=name, sf=scale_factor, p=precision: self._update_param_from_slider(n, value / sf, p))
            param_layout.addWidget(slider)

            value_label = QLabel(f"{self.simulator.params[name]:.{precision}f}")
            value_label.setMinimumWidth(50)
            param_layout.addWidget(value_label)

            self.control_layout.addLayout(param_layout)
            self.sliders[name] = {'slider': slider, 'label': value_label, 'scale': scale_factor, 'precision': precision}

        self.control_layout.addStretch()

    def _get_slider_config(self, name):
        """Return (min, max, scale_factor, precision) for sliders."""
        if name == 'dt': return 0.01, 0.2, 1000, 3
        if name == 'damping': return 0.0, 0.01, 10000, 4
        if name == 'base_c_sq': return 0.1, 5.0, 100, 2
        if name == 'tension_factor': return 0.0, 20.0, 10, 1
        if name == 'potential_lin': return 0.1, 2.0, 100, 2
        if name == 'potential_cub': return 0.0, 1.0, 100, 2
        if name == 'biharmonic_gamma': return 0.0, 0.1, 10000, 4
        return 0.0, 1.0, 100, 2 # Default

    def _update_param_from_slider(self, name, value, precision):
        """Update simulator parameter and label when slider moves."""
        self.simulator.update_params({name: value})
        self.sliders[name]['label'].setText(f"{value:.{precision}f}")

    def _setup_pyvista_scene(self):
        """Initialize the PyVista plotter scene."""
        print("Setting up PyVista scene...")
        phi_data = self.simulator.get_phi_copy()

        self.pv_grid = pv.ImageData()
        self.pv_grid.dimensions = np.array(phi_data.shape) + 1
        self.pv_grid.cell_data['phi'] = phi_data.ravel(order='F')
        self.pv_grid_pdata = self.pv_grid.cell_data_to_point_data()

        iso_pos = 1.5
        iso_neg = -1.0

        print(f"Adding initial isosurfaces at {iso_pos:.2f} and {iso_neg:.2f}")
        try:
            # Use grid_pdata for contouring
            isosurfaces = self.pv_grid_pdata.contour(isosurfaces=[iso_pos, iso_neg], scalars='phi', preference='points', compute_normals=True)
            if isosurfaces.n_points > 0:
                 self.plotter.add_mesh(isosurfaces, name='phi_isosurfaces', scalars='phi', cmap='coolwarm',
                                       opacity=0.6, show_scalar_bar=False)
            else:
                 print("Warning: Initial isosurfaces are empty.")
        except Exception as e:
             print(f"Warning: Could not create initial isosurfaces: {e}")

        self.plotter.add_mesh(self.pv_grid.outline(), color='grey')
        self.plotter.camera_position = 'xy'
        self.plotter.reset_camera()
        print("PyVista scene setup complete.")

    def _update_visualization(self):
        """Update the PyVista plot with current simulation data."""
        if not self.is_simulation_running: return

        phi_data = self.simulator.get_phi_copy()

        # Update CELL data
        self.pv_grid.cell_data['phi'] = phi_data.ravel(order='F')

        # RECALCULATE point data
        try:
            self.pv_grid_pdata = self.pv_grid.cell_data_to_point_data()
            if 'phi' not in self.pv_grid_pdata.point_data:
                # print("Warning: Point data conversion failed during update.") # Can be noisy
                return
        except Exception as e_conv:
            # print(f"Error during cell_data_to_point_data update: {e_conv}") # Can be noisy
            return

        # Update visualization actor
        try:
            iso_pos = 2 # Fixed example
            iso_neg = -2.5 # Fixed example

            if 'phi' not in self.pv_grid_pdata.point_data:
                 # print("Skipping contour: Point data 'phi' not found.") # Can be noisy
                 return

            new_isosurfaces = self.pv_grid_pdata.contour(isosurfaces=[iso_pos, iso_neg], scalars='phi', preference='points', compute_normals=True)

            # Robust Actor Update (Remove/Add)
            self.plotter.remove_actor('phi_isosurfaces', render=False)
            if new_isosurfaces.n_points > 0 and new_isosurfaces.faces.size > 0:
                 self.plotter.add_mesh(new_isosurfaces, name='phi_isosurfaces', scalars='phi', cmap='coolwarm',
                                       opacity=0.6, show_scalar_bar=False)
            # else: # Optional: print warning if contour is empty/non-polygonal
            #      if self.simulator.step_count % 50 == 0:
            #           print("Warning: Contour generated non-polygonal data or was empty. Skipping mesh add.")

        except Exception as e:
            if self.simulator.step_count % 50 == 0:
                 print(f"Error updating isosurfaces/GUI at step {self.simulator.step_count}: {e}")

        # Use processEvents to keep GUI responsive
        QApplication.processEvents()

    def _start_simulation(self):
        """Start the simulation thread."""
        if self.sim_thread is None or not self.sim_thread.isRunning():
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.reset_button.setEnabled(False)
            self.is_simulation_running = True

            for name, slider_info in self.sliders.items():
                 value = slider_info['slider'].value() / slider_info['scale']
                 self.simulator.update_params({name: value})

            self.sim_worker = SimulationWorker(self.simulator)
            self.sim_thread = QThread()
            self.sim_worker.moveToThread(self.sim_thread)
            self.sim_thread.started.connect(self.sim_worker.run)
            self.sim_worker.finished.connect(self._on_sim_thread_finished)
            self.sim_thread.start()
            self.gui_update_timer.start()
            print("Simulation started via GUI.")

    def _stop_simulation(self):
        """Stop the simulation thread."""
        self.is_simulation_running = False
        if self.sim_worker: self.sim_worker.stop()
        if self.sim_thread:
             self.sim_thread.quit()
             if not self.sim_thread.wait(1000):
                  print("Warning: Simulation thread did not stop cleanly. Terminating.")
                  self.sim_thread.terminate()
        self.gui_update_timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        print("Simulation stopped via GUI.")

    def _on_sim_thread_finished(self):
        """Handle cleanup when simulation thread finishes."""
        print("Simulation thread finished signal received.")
        self.is_simulation_running = False
        self.gui_update_timer.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.reset_button.setEnabled(True)
        self.sim_thread = None
        self.sim_worker = None

    def _reset_simulation(self):
        """Reset the simulation state."""
        if self.is_simulation_running:
            print("Warning: Cannot reset while simulation is running. Stop first.")
            return
        print("Resetting simulation...")
        # Update parameters from GUI before resetting
        for name, slider_info in self.sliders.items():
            value = slider_info['slider'].value() / slider_info['scale']
            self.simulator.update_params({name: value})

        # <<< --- CALL THE CORRECT METHOD --- >>>
        self.simulator.reset_simulation()
        # <<< --- END CORRECTION --- >>>

        # Update PyVista grid and visualization immediately
        self.pv_grid.cell_data['phi'] = self.simulator.phi.ravel(order='F')
        self.pv_grid_pdata = self.pv_grid.cell_data_to_point_data()
        self._update_visualization() # Update visualization state
        self.plotter.render() # Force render after reset
        print("Simulation reset complete.")

    def closeEvent(self, event):
        """Handle window close event."""
        self._stop_simulation()
        print("Closing application window.")
        self.plotter.close() # Close PyVista plotter window
        super().closeEvent(event)


# --- Main Application Execution ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    sim = EmergentParticleSimulator3D(grid_size=64)
    window = MainWindow(sim)
    window.show()
    sys.exit(app.exec_())

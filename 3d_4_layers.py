# -*- coding: utf-8 -*-
import sys, time, threading
import numpy as np
from scipy.ndimage import convolve
from numpy.fft import fftn, ifftn, fftfreq, fftshift # Keep for potential future analysis

try:
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QSlider, QPushButton, QFrame, QMessageBox, QRadioButton,
        QGroupBox, QSpinBox, QDoubleSpinBox # Added SpinBox for some params
    )
    import pyvista as pv
    from pyvistaqt import QtInteractor
except ImportError as e:
    print("ERROR: Required libraries (PyQt5, pyvista, pyvistaqt) not found.")
    print("Please install them: pip install numpy scipy PyQt5 pyvista pyvistaqt")
    print(f"Import Error: {e}")
    sys.exit(1)

# --- Simulation Engine ---
class ExplicitWoWSimulator3D:
    """
    N-layer, coupled 3D scalar-field simulator following TADS/WoW/Clockfield principles.
    All layers use the same grid size for visualization simplicity.
    """
    def __init__(self,
                 num_layers: int = 4,
                 grid_size: int = 32, # Single grid size for all layers
                 m0: float = 0.1,     # Base mass (Lower default)
                 alpha: float = 2.0,  # Mass scaling
                 c0: float = 1.0,     # Base speed
                 gamma: float = 1.5,  # Speed scaling (Set >= 1)
                 beta: float = 0.7,   # Coupling decay
                 xi: float = 0.1,     # Coupling strength
                 dt: float = 0.04,
                 damping: float = 0.001,
                 tension0: float = 5.0, # Layer 0 TADS tension
                 potential_lin0: float = 1.0, # Layer 0 potential
                 potential_cub0: float = 0.2): # Layer 0 potential

        print(f"Initializing {num_layers}-Layer WoW Simulation on {grid_size}^3 grid.")
        self.N = num_layers
        self.grid_size = grid_size
        self.lock = threading.Lock()

        # --- Store parameters directly ---
        self.params = {
             'dt': dt, 'damping': damping, 'm0': m0, 'alpha': alpha,
             'c0': c0, 'gamma': gamma, 'beta': beta, 'xi': xi,
             'tension0': tension0, 'potential_lin0': potential_lin0,
             'potential_cub0': potential_cub0
        }
        self.update_interval_ms = 50 # Target GUI update interval

        # --- Precompute layer-specific constants ---
        self._update_layer_constants()

        # --- Allocate fields ---
        self.phi = [np.zeros((grid_size,)*3, dtype=np.float64) for _ in range(self.N)]
        self.phi_old = [np.zeros_like(f) for f in self.phi]
        self.t = 0.0
        self.step_count = 0

        # --- 3D Laplacian Kernel ---
        k = np.zeros((3,3,3), np.float64); k[1,1,1] = -6
        for dx,dy,dz in [(1,1,0),(1,1,2),(1,0,1),(1,2,1),(0,1,1),(2,1,1)]: k[dx,dy,dz] = 1
        self.kern = k

        self.initialize_field()

    def _update_layer_constants(self):
        """Recalculate layer masses and speeds based on params."""
        m0_f = float(self.params['m0'])
        alpha_f = float(self.params['alpha'])
        c0_f = float(self.params['c0'])
        gamma_f = float(self.params['gamma'])
        self.m2 = [(m0_f * (alpha_f**n))**2 for n in range(self.N)]
        self.c2 = [(c0_f * (gamma_f**n))**2 for n in range(self.N)]
        print(f"Updated layer constants: m^2={self.m2}, c^2={self.c2}")

    def initialize_field(self):
        """Initialize all layers."""
        print("Initializing fields...")
        with self.lock:
            N = self.grid_size
            x, y, z = [np.arange(N)] * 3
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            cx = cy = cz = N // 2
            r2 = max((N / 8.0)**2, 1e-6)
            pulse = 2.0 * np.exp(-((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / (2 * r2))
            for n in range(self.N):
                 if n == 0: self.phi[n][:] = pulse
                 else: self.phi[n][:] = (np.random.rand(N,N,N) - 0.5) * 0.01
                 self.phi_old[n][:] = self.phi[n]
            self.t = 0.0; self.step_count = 0
        print("Fields Initialized.")

    def step(self):
        """Advance all layers by dt using Verlet + coupling."""
        with self.lock:
            new_phi_list = [None] * self.N
            for n in range(self.N):
                phi_n = self.phi[n]; phi_old_n = self.phi_old[n]
                lap = convolve(phi_n, self.kern, mode='wrap')
                if n == 0:
                    Vp = (-self.params['potential_lin0'] * phi_n + self.params['potential_cub0'] * (phi_n**3))
                    c2_loc = self.params['c0']**2 / (1.0 + self.params['tension0']*(phi_n**2) + 1e-9)
                    accel_intrinsic = c2_loc * lap - Vp
                else:
                    accel_intrinsic = self.c2[n] * lap - self.m2[n] * phi_n
                coup = np.zeros_like(phi_n)
                beta_n = self.params['beta']**n
                xi_eff = self.params['xi'] * beta_n
                if n > 0: coup += xi_eff * (self.phi[n-1] - phi_n)
                if n < self.N - 1: coup += xi_eff * (self.phi[n+1] - phi_n)
                vel = phi_n - phi_old_n
                phi_new = phi_n + (1.0 - self.params['damping']*self.params['dt'])*vel \
                          + (self.params['dt']**2)*(accel_intrinsic + coup)
                new_phi_list[n] = phi_new
            for n in range(self.N):
                self.phi_old[n][:] = self.phi[n]
                self.phi[n][:] = new_phi_list[n]
            self.t += self.params['dt']; self.step_count += 1

    def get_field_copy(self, n: int) -> np.ndarray:
        """Return a thread-safe copy of the current field for layer n."""
        if 0 <= n < self.N:
            with self.lock: return self.phi[n].copy()
        else: print(f"Error: Invalid layer index {n}"); return np.zeros((self.grid_size,)*3, dtype=np.float64)

    def update_params(self, d):
        """Update parameters dictionary (thread-safe)."""
        with self.lock:
            params_changed = False
            for k, v in d.items():
                if k in self.params:
                    try:
                        new_val = float(v)
                        if self.params[k] != new_val: self.params[k] = new_val; params_changed = True
                    except (ValueError, TypeError): print(f"Warning: Invalid value type for {k}: {v}")
                else: print(f"Warning: Attempted to update unknown parameter: {k}")
            if params_changed and any(k in ['m0', 'alpha', 'c0', 'gamma'] for k in d): self._update_layer_constants()

    def reset_simulation(self): self.initialize_field()

# --- Simulation Thread ---
class SimulationWorker(QObject):
    finished = pyqtSignal()
    def __init__(self, sim): super().__init__(); self.sim = sim; self._running = False
    def run(self):
        self._running = True; print("Simulation thread started.")
        while self._running:
            t0 = time.perf_counter()
            self.sim.step()
            dt = (time.perf_counter()-t0)*1000
            to_sleep = max(1, self.sim.update_interval_ms - dt)/1000.0
            time.sleep(to_sleep)
        print("Simulation thread finished."); self.finished.emit()
    def stop(self): print("Stopping simulation thread..."); self._running = False

# --- Main Window ---
class MainWindow(QMainWindow):
    """GUI for Explicit WoW Simulation, visualizing one layer at a time."""
    def __init__(self, sim_engine):
        super().__init__()
        self.sim = sim_engine
        self.setWindowTitle("Interactive 3D Explicit WoW Simulation")
        self.resize(1300, 800)

        cw = QWidget(); self.setCentralWidget(cw)
        main_layout = QHBoxLayout(cw)
        pf = QFrame(); pf.setFrameStyle(QFrame.StyledPanel|QFrame.Sunken)
        pv_layout = QVBoxLayout(pf); self.plotter = QtInteractor(pf)
        pv_layout.addWidget(self.plotter); main_layout.addWidget(pf, stretch=3)
        ctrl = QWidget(); ctrl.setFixedWidth(400)
        cl = QVBoxLayout(ctrl); main_layout.addWidget(ctrl, stretch=1)

        self.vis_layer_index = 0
        self.iso_value = 1.0

        self._setup_controls(cl)
        self._scene_setup()

        self.worker = None; self.thread = None
        self.timer = QTimer(self); self.timer.timeout.connect(self._refresh)
        self.timer.setInterval(self.sim.update_interval_ms)

    def _slider_conf(self, n):
        """Slider configurations."""
        if n == 'dt': return 0.01, 0.1, 1000, 3
        if n == 'damping': return 0, 0.01, 10000, 4
        if n == 'm0': return 0.01, 2.0, 100, 2
        if n == 'alpha': return 1.0, 3.0, 100, 2
        if n == 'c0': return 0.1, 5.0, 100, 2
        if n == 'gamma': return 1.0, 3.0, 100, 2
        if n == 'beta': return 0.1, 1.0, 100, 2
        if n == 'xi': return 0.0, 1.0, 100, 2
        if n == 'tension0': return 0, 20, 10, 1
        if n == 'potential_lin0': return 0.1, 2, 100, 2
        if n == 'potential_cub0': return 0, 1, 100, 2
        return 0, 1, 100, 2

    def _setup_controls(self, cl):
        """Create GUI controls."""
        # Start/Stop/Reset
        hl = QHBoxLayout()
        self.start_b = QPushButton("Start"); self.start_b.clicked.connect(self.start)
        self.stop_b = QPushButton("Stop"); self.stop_b.clicked.connect(self.stop); self.stop_b.setEnabled(False)
        self.reset_b = QPushButton("Reset"); self.reset_b.clicked.connect(self.reset)
        hl.addWidget(self.start_b); hl.addWidget(self.stop_b); hl.addWidget(self.reset_b)
        cl.addLayout(hl)

        # Parameter Sliders/Spinboxes
        cl.addWidget(QLabel("Simulation Parameters:"))
        self.param_controls = {}
        for name in self.sim.params:
            h = QHBoxLayout()
            lbl_name = QLabel(name.replace('_', ' ').title()); lbl_name.setMinimumWidth(110)
            if name in ['dt', 'damping', 'beta', 'xi', 'm0', 'alpha', 'gamma']: # Use SpinBox for these
                 spin = QDoubleSpinBox()
                 mn, mx, _, pr = self._slider_conf(name)
                 spin.setRange(mn, mx); spin.setDecimals(pr+1); spin.setSingleStep(10**(-pr))
                 spin.setValue(self.sim.params[name])
                 spin.valueChanged.connect(lambda v, n=name, p=pr: self._param_changed(n, v, p))
                 widget = spin
            else: # Use sliders for others
                 sld = QSlider(Qt.Horizontal)
                 mn, mx, sc, pr = self._slider_conf(name)
                 sld.setRange(int(mn * sc), int(mx * sc))
                 sld.setValue(int(self.sim.params[name] * sc))
                 sld.valueChanged.connect(lambda v, n=name, s=sc, p=pr: self._param_changed(n, v / s, p))
                 widget = sld
            val_lbl = QLabel(f"{self.sim.params[name]:.{pr}f}")
            val_lbl.setMinimumWidth(60)
            h.addWidget(lbl_name); h.addWidget(widget); h.addWidget(val_lbl)
            cl.addLayout(h)
            self.param_controls[name] = {'w': widget, 'l': val_lbl, 'pr': pr}
            if isinstance(widget, QSlider): self.param_controls[name]['sc'] = sc

        # Visualization Controls
        vis_group = QGroupBox("Visualization")
        gl = QVBoxLayout(vis_group)
        layer_sel_layout = QHBoxLayout(); layer_sel_layout.addWidget(QLabel("Visualize Layer:"))
        self.vis_layer_buttons = []
        for i in range(self.sim.N):
             rb = QRadioButton(f"{i}"); rb.setChecked(i == self.vis_layer_index)
             rb.toggled.connect(lambda checked, idx=i: self._vis_layer_changed(idx, checked))
             layer_sel_layout.addWidget(rb); self.vis_layer_buttons.append(rb)
        layer_sel_layout.addStretch(); gl.addLayout(layer_sel_layout)
        iso_layout = QHBoxLayout(); iso_label = QLabel("Iso Value:")
        self.iso_slider = QSlider(Qt.Horizontal)
        self.iso_slider.setRange(-300, 300); self.iso_slider.setValue(int(self.iso_value * 100))
        self.iso_value_label = QLabel(f"{self.iso_value:.2f}"); self.iso_value_label.setMinimumWidth(40)
        self.iso_slider.valueChanged.connect(lambda v: self._iso_changed(v / 100.0))
        iso_layout.addWidget(iso_label); iso_layout.addWidget(self.iso_slider); iso_layout.addWidget(self.iso_value_label)
        gl.addLayout(iso_layout)
        cl.addWidget(vis_group)
        cl.addStretch()

    def _param_changed(self, name, value, precision):
        """Update simulator parameter and GUI label."""
        self.sim.update_params({name: value})
        # Update label immediately
        self.param_controls[name]['l'].setText(f"{value:.{precision}f}")


    def _vis_layer_changed(self, index, checked):
        """Handle radio button selection for visualized layer."""
        if checked:
             self.vis_layer_index = index
             print(f"Switched visualization to Layer {index}")
             if not (self.worker and self.worker._running): self._draw_once()

    def _iso_changed(self, value):
        """Update stored iso value and GUI label."""
        self.iso_value = value
        self.iso_value_label.setText(f"{value:.2f}")
        if not (self.worker and self.worker._running): self._draw_once()

    def _scene_setup(self):
        """Setup initial PyVista scene."""
        self.plotter.clear_actors()
        self.plotter.add_axes()
        # Create the PyVista Grid object for visualization updates
        phi_data = self.sim.get_field_copy(self.vis_layer_index)
        self.pv_grid = pv.ImageData()
        self.pv_grid.dimensions = np.array(phi_data.shape) + 1
        self.plotter.add_mesh(self.pv_grid.outline(), color='grey', name='outline')
        # Initial draw
        self._draw_once()
        self.plotter.camera_position = 'xy'
        self.plotter.reset_camera()
        print("PyVista scene setup complete.")

    def _draw_once(self):
        """Fetch data for selected layer and update PyVista plot."""
        phi_n = self.sim.get_field_copy(self.vis_layer_index)

        # Update or create the grid object
        # <<< --- FIX: Correct variable name used in comparison --- >>>
        if not hasattr(self, 'pv_grid') or \
           (not np.array_equal(self.pv_grid.dimensions, np.array(phi_n.shape) + 1)):
        # <<< --- END FIX --- >>>
            self.pv_grid = pv.ImageData(dimensions=np.array(phi_n.shape) + 1)
            # print("Created/Resized vis_grid.") # Less verbose

        # Assign cell data and convert to point data
        self.pv_grid.cell_data['phi'] = phi_n.ravel(order='F')
        vis_grid_pdata = self.pv_grid.cell_data_to_point_data()

        if 'phi' not in vis_grid_pdata.point_data:
            # print(f"Warning: Point data failed for layer {self.vis_layer_index}") # Less verbose
            return

        # Remove old isosurface actor
        actor_name = f'layer_{self.vis_layer_index}_iso'
        self.plotter.remove_actor(actor_name, render=False)

        # Create and add new isosurface
        try:
            iso_val = self.iso_value
            mesh = vis_grid_pdata.contour([iso_val], scalars='phi', preference='points')
            if mesh.n_points > 0 and mesh.n_faces > 0:
                # Color by the actual phi value
                self.plotter.add_mesh(mesh, name=actor_name, scalars='phi', cmap='coolwarm',
                                      opacity=0.6, show_scalar_bar=True, scalar_bar_args={'title': f'Phi Layer {self.vis_layer_index}'})
            # else: # Don't print warning every time contour is empty
                 # if self.sim.step_count % 100 == 0: print(f"No surface at iso={iso_val:.2f} for layer {self.vis_layer_index}")
        except Exception as e_draw:
            if self.sim.step_count % 50 == 0:
                print(f"Error drawing layer {self.vis_layer_index}: {e_draw}")

        self.plotter.render()

    def _refresh(self):
        """Timer callback to refresh visualization."""
        if self.worker and self.worker._running:
            self._draw_once()

    def start(self):
        """Start simulation thread and GUI timer."""
        if not (self.worker and self.worker._running):
             # Update sim params from GUI before starting
             for name, ctrl_info in self.param_controls.items():
                  if isinstance(ctrl_info['w'], QSlider): value = ctrl_info['w'].value() / ctrl_info['sc']
                  else: value = ctrl_info['w'].value()
                  self.sim.update_params({name: value})

             self.worker = SimulationWorker(self.sim); self.thread = QThread()
             self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run)
             self.worker.finished.connect(self.thread.quit); self.worker.finished.connect(self.worker.deleteLater)
             self.thread.finished.connect(self.thread.deleteLater); self.thread.start()
             self.timer.start()
             self.start_b.setEnabled(False); self.stop_b.setEnabled(True); self.reset_b.setEnabled(False)

    def stop(self):
        """Stop simulation thread and GUI timer."""
        if self.worker: self.worker.stop()
        self.timer.stop()
        if self.thread and self.thread.isRunning():
             self.thread.quit()
             if not self.thread.wait(500): self.thread.terminate()
        self.start_b.setEnabled(True); self.stop_b.setEnabled(False); self.reset_b.setEnabled(True)

    def reset(self):
        """Stop simulation, reset state, and redraw."""
        was_running = (self.worker and self.worker._running)
        if was_running: self.stop()
        # Update params from GUI before resetting
        for name, ctrl_info in self.param_controls.items():
             if isinstance(ctrl_info['w'], QSlider): value = ctrl_info['w'].value() / ctrl_info['sc']
             else: value = ctrl_info['w'].value()
             self.sim.update_params({name: value})
        self.sim.reset_simulation()
        self._draw_once() # Draw the reset state

    def closeEvent(self, ev):
        """Handle window close event."""
        self.stop()
        self.plotter.close()
        super().closeEvent(ev)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Initialize with default parameters, start with smaller grid for performance
    sim = ExplicitWoWSimulator3D(num_layers=4, grid_size=32)
    w = MainWindow(sim)
    w.show()
    sys.exit(app.exec_())

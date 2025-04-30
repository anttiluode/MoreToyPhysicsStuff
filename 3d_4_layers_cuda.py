# -*- coding: utf-8 -*-
import sys, time, threading
# Use CuPy for GPU arrays and operations
try:
    import cupy as cp
    # Attempt to import the CuPy equivalent for convolve
    from cupyx.scipy.ndimage import convolve as cp_convolve
    # Keep numpy for CPU-specific things if needed, like meshgrid indices initially
    import numpy as np
    print("CuPy found. Attempting GPU acceleration.")
    GPU_ENABLED = True
except ImportError:
    print("CuPy or cupyx.scipy.ndimage not found. Falling back to NumPy on CPU.")
    import numpy as cp # Use numpy as alias if cupy fails
    from scipy.ndimage import convolve as cp_convolve # Use scipy convolve
    import numpy as np
    GPU_ENABLED = False

from numpy.fft import fftn, ifftn, fftfreq, fftshift # Keep numpy FFT for analysis part

try:
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QSlider, QPushButton, QFrame, QMessageBox, QRadioButton,
        QGroupBox, QSpinBox, QDoubleSpinBox
    )
    import pyvista as pv
    from pyvistaqt import QtInteractor
except ImportError as e:
    print("ERROR: Required GUI/Visualization libraries (PyQt5, pyvista, pyvistaqt) not found.")
    print("Please install them: pip install PyQt5 pyvista pyvistaqt")
    print(f"Import Error: {e}")
    sys.exit(1)

# --- Simulation Engine (Modified for CuPy) ---
class ExplicitWoWSimulator3D_GPU:
    """
    N-layer, coupled 3D scalar-field simulator using CuPy for GPU acceleration.
    """
    def __init__(self,
                 num_layers: int = 4,
                 grid_size: int = 64, # Keep default reasonable
                 m0: float = 0.1, alpha: float = 2.0,
                 c0: float = 1.0, gamma: float = 1.5,
                 beta: float = 0.7, xi: float = 0.1,
                 dt: float = 0.04, damping: float = 0.001,
                 tension0: float = 5.0, potential_lin0: float = 1.0,
                 potential_cub0: float = 0.2):

        print(f"Initializing {num_layers}-Layer WoW Simulation on {grid_size}^3 grid (Device: {'GPU' if GPU_ENABLED else 'CPU'}).")
        self.N = num_layers
        self.grid_size = grid_size
        self.lock = threading.Lock() # Keep thread lock for potential GUI interaction

        # --- Store parameters ---
        # Use standard floats for parameters, CuPy arrays for fields
        self.params = {
            'dt': dt, 'damping': damping, 'm0': m0, 'alpha': alpha,
            'c0': c0, 'gamma': gamma, 'beta': beta, 'xi': xi,
            'tension0': tension0, 'potential_lin0': potential_lin0,
            'potential_cub0': potential_cub0
        }
        self.update_interval_ms = 50

        # --- Precompute layer-specific constants (as standard Python floats/lists) ---
        self._update_layer_constants()

        # --- Allocate fields on the selected device (GPU or CPU) ---
        print("Allocating fields...")
        xp = cp # Use the alias (cp will be cupy or numpy)
        self.phi = [xp.zeros((grid_size,)*3, dtype=xp.float64) for _ in range(self.N)]
        self.phi_old = [xp.zeros_like(f) for f in self.phi]
        print(f"Field array type: {type(self.phi[0])}")

        self.t = 0.0
        self.step_count = 0

        # --- 3D Laplacian Kernel on the selected device ---
        k_np = np.zeros((3,3,3), np.float64); k_np[1,1,1] = -6
        for dx,dy,dz in [(1,1,0),(1,1,2),(1,0,1),(1,2,1),(0,1,1),(2,1,1)]: k_np[dx,dy,dz] = 1
        self.kern = xp.asarray(k_np) # Convert kernel to CuPy/NumPy array

        self.initialize_field()

    def _update_layer_constants(self):
        """Recalculate layer masses and speeds based on params."""
        m0_f = float(self.params['m0'])
        alpha_f = float(self.params['alpha'])
        c0_f = float(self.params['c0'])
        gamma_f = float(self.params['gamma'])
        # Store m2 and c2 as regular Python lists of floats
        self.m2 = [(m0_f * (alpha_f**n))**2 for n in range(self.N)]
        self.c2 = [(c0_f * (gamma_f**n))**2 for n in range(self.N)]
        print(f"Updated layer constants: m^2={self.m2}, c^2={self.c2}")

    def initialize_field(self):
        """Initialize all layers."""
        print("Initializing fields...")
        xp = cp # Use alias
        with self.lock:
            N = self.grid_size
            # Use numpy for meshgrid indices initially, then convert if needed
            x_np, y_np, z_np = [np.arange(N)] * 3
            X_np, Y_np, Z_np = np.meshgrid(x_np, y_np, z_np, indexing='ij')
            cx = cy = cz = N // 2
            r2 = max((N / 8.0)**2, 1e-6)

            # Calculate pulse on CPU with numpy
            pulse_np = 2.0 * np.exp(-((X_np - cx)**2 + (Y_np - cy)**2 + (Z_np - cz)**2) / (2 * r2))
            # Calculate random noise on CPU with numpy
            rand_noise_np = (np.random.rand(N, N, N) - 0.5) * 0.01

            # Assign to fields, converting to CuPy arrays
            for n in range(self.N):
                if n == 0:
                    self.phi[n][:] = xp.asarray(pulse_np)
                else:
                    self.phi[n][:] = xp.asarray(rand_noise_np) # Use same random noise for layers > 0
                self.phi_old[n][:] = self.phi[n] # Copy on device

            self.t = 0.0
            self.step_count = 0
            # Ensure memory operations are complete if using GPU streams
            if GPU_ENABLED:
                cp.cuda.Stream.null.synchronize()
        print("Fields Initialized.")

    def step(self):
        """Advance all layers by dt using Verlet + coupling on GPU/CPU."""
        xp = cp # Use alias
        dt = self.params['dt'] # Get float parameters
        damping = self.params['damping']
        xi = self.params['xi']
        beta = self.params['beta']
        tension0 = self.params['tension0']
        potential_lin0 = self.params['potential_lin0']
        potential_cub0 = self.params['potential_cub0']
        c0_sq = self.c2[0] # Base speed squared from list

        # Lock might not be strictly necessary if GUI only reads via get_field_copy
        # but keep for safety if params can be updated live
        with self.lock:
            new_phi_list = [None] * self.N # Use Python list to store results temporarily

            for n in range(self.N):
                phi_n = self.phi[n]; phi_old_n = self.phi_old[n]

                # Use cp_convolve (cupyx.scipy.ndimage or scipy.ndimage)
                lap = cp_convolve(phi_n, self.kern, mode='wrap')

                # Layer-specific physics
                if n == 0:
                    Vp = (-potential_lin0 * phi_n + potential_cub0 * (phi_n**3))
                    c2_loc = c0_sq / (1.0 + tension0*(phi_n**2) + 1e-9)
                    accel_intrinsic = c2_loc * lap - Vp
                else:
                    # Use precomputed Python floats for m2, c2 for this layer
                    c2_n = self.c2[n]
                    m2_n = self.m2[n]
                    accel_intrinsic = c2_n * lap - m2_n * phi_n

                # Coupling term (needs values from other layers)
                coup = xp.zeros_like(phi_n)
                beta_n = beta**n # Python float calculation
                xi_eff = xi * beta_n # Python float calculation

                if n > 0:
                    # Ensure compatible types if mixing numpy/cupy (shouldn't happen here)
                    coup += xi_eff * (self.phi[n-1] - phi_n)
                if n < self.N - 1:
                    coup += xi_eff * (self.phi[n+1] - phi_n)

                # Verlet update step
                vel = phi_n - phi_old_n
                phi_new = phi_n + (1.0 - damping * dt) * vel \
                          + (dt**2) * (accel_intrinsic + coup)

                new_phi_list[n] = phi_new # Store the new CuPy/NumPy array

            # Update fields after calculating all new states
            for n in range(self.N):
                self.phi_old[n][:] = self.phi[n] # Copy happens on device
                self.phi[n][:] = new_phi_list[n] # Assign new state on device

            self.t += dt
            self.step_count += 1

            # Ensure calculations are done if using GPU streams (optional, default stream syncs)
            if GPU_ENABLED:
                cp.cuda.Stream.null.synchronize()


    def get_field_copy(self, n: int) -> np.ndarray:
        """
        Return a thread-safe *NumPy* copy of the current field for layer n
        for visualization or analysis on the CPU.
        """
        if 0 <= n < self.N:
            with self.lock:
                # Copy from GPU to CPU using cp.asnumpy() if using CuPy
                field_device = self.phi[n]
                if GPU_ENABLED:
                    return cp.asnumpy(field_device)
                else:
                    return np.copy(field_device) # If already numpy, just copy
        else:
            print(f"Error: Invalid layer index {n}")
            # Return numpy array for consistency
            return np.zeros((self.grid_size,) * 3, dtype=np.float64)

    def update_params(self, d):
        """Update parameters dictionary (thread-safe)."""
        # No CuPy needed here, just updating Python dict
        with self.lock:
            params_changed = False
            for k, v in d.items():
                if k in self.params:
                    try:
                        new_val = float(v)
                        if self.params[k] != new_val:
                            self.params[k] = new_val
                            params_changed = True
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid value type for {k}: {v}")
                else:
                    print(f"Warning: Attempted to update unknown parameter: {k}")
            # If scaling params changed, update derived constants
            if params_changed and any(k in ['m0', 'alpha', 'c0', 'gamma'] for k in d):
                self._update_layer_constants()

    def reset_simulation(self):
        """Reset simulation state."""
        # This will re-initialize fields on the GPU/CPU
        self.initialize_field()

# --- Simulation Thread (Unchanged) ---
# (SimulationWorker class code remains the same as before)
class SimulationWorker(QObject):
    # ... (same as before) ...
    finished = pyqtSignal()
    def __init__(self, sim): super().__init__(); self.sim = sim; self._running = False
    def run(self):
        self._running = True; print("Simulation thread started.")
        while self._running:
            t0 = time.perf_counter()
            self.sim.step()
            dt_calc = (time.perf_counter()-t0)*1000
            # Use the simulation's interval target
            to_sleep = max(1, self.sim.update_interval_ms - dt_calc)/1000.0
            if self._running: # Check again before sleeping
                 time.sleep(to_sleep)
        print("Simulation thread finished."); self.finished.emit()
    def stop(self): print("Stopping simulation thread..."); self._running = False

# --- Main Window (Mostly Unchanged GUI, Updated Sim Init) ---
# (MainWindow class code remains mostly the same, but needs to
#  instantiate the correct simulator class and handle potential
#  differences in data types if any arise, although get_field_copy
#  ensures numpy arrays for visualization)
class MainWindow(QMainWindow):
    # ... ( __init__ , _slider_conf, _setup_controls, _param_changed,
    #        _vis_layer_changed, _iso_changed remain largely the same) ...
    def __init__(self, sim_engine):
        super().__init__()
        self.sim = sim_engine # Now expects ExplicitWoWSimulator3D_GPU
        self.setWindowTitle("Interactive 3D Explicit WoW Simulation (GPU Attempt)")
        self.resize(1300, 800)

        # Layout setup (same as before)
        cw = QWidget(); self.setCentralWidget(cw)
        main_layout = QHBoxLayout(cw)
        pf = QFrame(); pf.setFrameStyle(QFrame.StyledPanel|QFrame.Sunken)
        pv_layout = QVBoxLayout(pf); self.plotter = QtInteractor(pf)
        pv_layout.addWidget(self.plotter); main_layout.addWidget(pf, stretch=3)
        ctrl = QWidget(); ctrl.setFixedWidth(400)
        cl = QVBoxLayout(ctrl); main_layout.addWidget(ctrl, stretch=1)

        self.vis_layer_index = 0
        self.iso_value = 1.0

        self._setup_controls(cl) # Controls setup uses sim.params
        self._scene_setup()

        self.worker = None; self.thread = None
        self.update_timer = QTimer(self) # Renamed for clarity
        self.update_timer.timeout.connect(self._refresh)
        self.update_timer.setInterval(self.sim.update_interval_ms) # Use sim's interval

    def _slider_conf(self, n):
        # (Same as before)
        if n=='dt': return 0.01,0.1,1000,3
        if n=='damping': return 0,0.01,10000,4
        if n=='m0': return 0.01, 5.0, 100, 2 # Increased m0 range slightly
        if n=='alpha': return 1.0, 4.0, 100, 2 # Increased alpha range slightly
        if n=='c0': return 0.1, 5.0, 100, 2
        if n=='gamma': return 1.0, 4.0, 100, 2 # Increased gamma range slightly
        if n=='beta': return 0.1, 1.0, 100, 2
        if n=='xi': return 0.0, 1.0, 100, 2
        if n=='tension0': return 0, 20, 10, 1
        if n=='potential_lin0': return 0.1, 2, 100, 2
        if n=='potential_cub0': return 0, 1, 100, 2
        return 0,1,100,2

    def _setup_controls(self, cl):
        # (Same setup using self.sim.params as before)
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
            # Decide between slider and spinbox (adjust as needed)
            use_spinbox = name in ['dt', 'damping', 'beta', 'xi']
            if use_spinbox:
                widget = QDoubleSpinBox()
                mn, mx, _, pr = self._slider_conf(name)
                widget.setRange(mn, mx); widget.setDecimals(pr+1); widget.setSingleStep(10**(-pr))
                widget.setValue(self.sim.params[name])
                # Connect valueChanged signal
                widget.valueChanged.connect(lambda v, n=name, p=pr: self._param_changed(n, v, p))
            else: # Use sliders for others
                widget = QSlider(Qt.Horizontal)
                mn, mx, sc, pr = self._slider_conf(name)
                widget.setRange(int(mn * sc), int(mx * sc))
                widget.setValue(int(self.sim.params[name] * sc))
                # Connect valueChanged signal
                widget.valueChanged.connect(lambda v, n=name, s=sc, p=pr: self._param_changed(n, v / s, p))

            val_lbl = QLabel(f"{self.sim.params[name]:.{pr}f}")
            val_lbl.setMinimumWidth(60)
            h.addWidget(lbl_name); h.addWidget(widget); h.addWidget(val_lbl)
            cl.addLayout(h)
            self.param_controls[name] = {'w': widget, 'l': val_lbl, 'pr': pr}
            if not use_spinbox: self.param_controls[name]['sc'] = sc # Store scale only for sliders

        # Visualization Controls
        vis_group = QGroupBox("Visualization")
        gl = QVBoxLayout(vis_group)
        layer_sel_layout = QHBoxLayout(); layer_sel_layout.addWidget(QLabel("Visualize Layer:"))
        self.vis_layer_buttons = []
        for i in range(self.sim.N): # Use self.sim.N
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
        # Update sim params (thread-safe method)
        self.sim.update_params({name: value})
        # Update label immediately
        # Check if control exists and update label
        if name in self.param_controls:
             self.param_controls[name]['l'].setText(f"{value:.{precision}f}")

    def _vis_layer_changed(self, index, checked):
        # (Same as before)
        if checked:
            self.vis_layer_index = index
            print(f"Switched visualization to Layer {index}")
            # Redraw immediately if simulation is stopped
            if not (self.worker and self.worker._running): self._draw_once()

    def _iso_changed(self, value):
        # (Same as before)
        self.iso_value = value
        self.iso_value_label.setText(f"{value:.2f}")
        if not (self.worker and self.worker._running): self._draw_once()


    def _scene_setup(self):
        # (Same as before, uses get_field_copy which handles CPU transfer)
        self.plotter.clear_actors()
        self.plotter.add_axes()
        # Get initial field (as numpy array)
        phi_data_np = self.sim.get_field_copy(self.vis_layer_index)
        if phi_data_np is None or phi_data_np.size == 0:
             print("Error: Could not get initial field data for scene setup.")
             return
        self.pv_grid = pv.ImageData()
        self.pv_grid.dimensions = np.array(phi_data_np.shape) + 1
        self.plotter.add_mesh(self.pv_grid.outline(), color='grey', name='outline')
        self._draw_once() # Initial draw
        self.plotter.camera_position = 'xy'
        self.plotter.reset_camera()
        print("PyVista scene setup complete.")

    def _draw_once(self):
        # (Same as before, uses get_field_copy which handles CPU transfer)
        phi_n_np = self.sim.get_field_copy(self.vis_layer_index)
        if phi_n_np is None or phi_n_np.size == 0:
             # print(f"Warning: No data returned for layer {self.vis_layer_index}")
             return # Skip drawing if data is invalid

        # Ensure grid exists and has correct dimensions
        try:
            target_dims = np.array(phi_n_np.shape) + 1
            if not hasattr(self, 'pv_grid') or not np.array_equal(self.pv_grid.dimensions, target_dims):
                self.pv_grid = pv.ImageData(dimensions=target_dims)
                # Re-add outline if grid is recreated? Maybe not necessary if only scalars change.
                if 'outline' not in self.plotter.actors:
                     self.plotter.add_mesh(self.pv_grid.outline(), color='grey', name='outline')

            # Assign cell data and convert to point data
            self.pv_grid.cell_data['phi'] = phi_n_np.ravel(order='F')
            vis_grid_pdata = self.pv_grid.cell_data_to_point_data()

            if 'phi' not in vis_grid_pdata.point_data:
                 print(f"Warning: Point data conversion failed for layer {self.vis_layer_index}")
                 return

            # Remove old isosurface actor
            actor_name = f'layer_{self.vis_layer_index}_iso'
            self.plotter.remove_actor(actor_name, render=False)

            # Create and add new isosurface
            iso_val = self.iso_value
            mesh = vis_grid_pdata.contour([iso_val], scalars='phi', preference='points')

            # Check mesh validity before adding
            if mesh.n_points > 0 and mesh.n_faces > 0:
                 self.plotter.add_mesh(mesh, name=actor_name, scalars='phi', cmap='coolwarm',
                                     opacity=0.6, show_scalar_bar=False) # Scalar bar off for speed?
                 # print(f"Layer {self.vis_layer_index} iso updated.") # Debug
            #else:
                 # if self.sim.step_count % 100 == 0: print(f"No surface at iso={iso_val:.2f} for layer {self.vis_layer_index}")

        except Exception as e_draw:
            # Add more context to error
            print(f"Error during drawing layer {self.vis_layer_index} at step {self.sim.step_count}: {e_draw}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging

        # Final render call
        self.plotter.render()


    def _refresh(self):
        # (Same as before)
        # Check if worker exists and is running before drawing
        if self.worker and hasattr(self.worker, '_running') and self.worker._running:
            self._draw_once()


    def start(self):
        # (Same as before, creates SimulationWorker which uses the GPU/CPU sim)
        if not (self.worker and hasattr(self.worker, '_running') and self.worker._running):
            # Update sim params from GUI before starting
            for name, ctrl_info in self.param_controls.items():
                if isinstance(ctrl_info['w'], QSlider): value = ctrl_info['w'].value() / ctrl_info['sc']
                else: value = ctrl_info['w'].value() # Assumes QDoubleSpinBox
                self.sim.update_params({name: value})

            self.worker = SimulationWorker(self.sim); self.thread = QThread()
            self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit); self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater); self.thread.start()
            self.update_timer.start() # Use renamed timer variable
            self.start_b.setEnabled(False); self.stop_b.setEnabled(True); self.reset_b.setEnabled(False)


    def stop(self):
        # (Same as before)
        if self.worker: self.worker.stop() # Signal worker to stop
        self.update_timer.stop() # Stop GUI updates
        # Wait for thread to finish cleanly
        if self.thread and self.thread.isRunning():
            self.thread.quit()
            if not self.thread.wait(1000): # Wait up to 1 sec
                 print("Warning: Simulation thread did not exit cleanly. Terminating.")
                 self.thread.terminate() # Force terminate if necessary
        self.worker = None # Clear worker/thread references
        self.thread = None
        self.start_b.setEnabled(True); self.stop_b.setEnabled(False); self.reset_b.setEnabled(True)
        print("Simulation stopped via GUI.")

    def reset(self):
        # (Same as before, ensures GUI params are passed before reset)
        was_running = (self.worker and hasattr(self.worker, '_running') and self.worker._running)
        if was_running: self.stop()
        # Update params from GUI before resetting
        for name, ctrl_info in self.param_controls.items():
            if isinstance(ctrl_info['w'], QSlider): value = ctrl_info['w'].value() / ctrl_info['sc']
            else: value = ctrl_info['w'].value()
            self.sim.update_params({name: value})
        self.sim.reset_simulation()
        self._draw_once() # Draw the reset state
        print("Simulation reset.")


    def closeEvent(self, ev):
        # (Same as before)
        print("Closing window...")
        self.stop()
        # self.plotter.close() # May not be necessary if parent window closes
        super().closeEvent(ev)




if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Initialize with the GPU/CPU simulator
    # Start with a smaller grid for interactive performance
    sim_instance = ExplicitWoWSimulator3D_GPU(num_layers=4, grid_size=32) # Use 32 or 64
    window = MainWindow(sim_instance)
    window.show()
    sys.exit(app.exec_())
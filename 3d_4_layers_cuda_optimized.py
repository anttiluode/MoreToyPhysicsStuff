# -*- coding: utf-8 -*-
"""
WoW multi‑layer explicit solver — FP16 storage / FP32 compute version

* Stores φ and φ_old in float16 to cut VRAM usage.
* All physics is computed in float32 for stability.
* Works on GPU via CuPy, falls back to NumPy on CPU.
* Uses a single scratch buffer `self.lap_f32` for Laplacian.
* Includes PyQt5/PyVista GUI for interactive control & visualization.
"""
import sys, time, threading, warnings
import traceback # For better error reporting

# -------------------------------------------------
# 1 ▸  Import CuPy if available, else NumPy fallback
# -------------------------------------------------
try:
    import cupy as cp
    try:
        from cupyx.scipy.ndimage import convolve as cp_convolve
    except ImportError:
        print("Warning: cupyx.scipy.ndimage.convolve not found. Will use SciPy via CPU.")
        from scipy.ndimage import convolve as cpu_convolve # Keep scipy convolve available
        # Define a wrapper that copies to CPU, convolves, copies back
        def cp_convolve_via_cpu(arr, kern, output=None, mode='wrap'): # Added output param
            arr_cpu = cp.asnumpy(arr)
            kern_cpu = cp.asnumpy(kern)
            result_cpu = cpu_convolve(arr_cpu, kern_cpu, mode=mode) # Output arg not standard for scipy
            result_gpu = cp.asarray(result_cpu, dtype=arr.dtype)
            if output is not None: output[:] = result_gpu # Write to output if provided
            else: return result_gpu
        cp_convolve = cp_convolve_via_cpu # Use the wrapper

    print("CuPy found – running on GPU (Convolution via CuPyX or CPU fallback)")
    GPU_ENABLED = True
except ImportError:
    import numpy as cp                        # alias NumPy → cp
    from scipy.ndimage import convolve as cp_convolve # Use SciPy convolve directly
    print("CuPy not found – running on CPU with NumPy")
    GPU_ENABLED = False

# Keep numpy for CPU tasks
import numpy as np
# from numpy.fft import fftn, ifftn, fftfreq, fftshift # Not used directly in this sim

# -------------------------------------------------
# 2 ▸  Qt + PyVista GUI imports
# -------------------------------------------------
try:
    from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QLabel, QSlider, QPushButton, QFrame,
                                 QGroupBox, QRadioButton, QDoubleSpinBox)
    import pyvista as pv
    from pyvistaqt import QtInteractor
    PYVISTA_AVAILABLE = True
except ImportError as e:
    print("ERROR: PyQt5 or PyVista/PyVistaQT are required for the GUI")
    print("-> pip install PyQt5 pyvista pyvistaqt")
    print(f"-> ImportError: {e}")
    # Allow script to run without GUI for basic testing if needed
    PYVISTA_AVAILABLE = False
    # --- Dummy classes if GUI fails to import ---
    # *** FIX: Correct Indentation Here ***
    class QMainWindow: pass
    class QObject: pass
    # Need to define pyqtSignal this way for older Qt versions sometimes
    class pyqtSignal:
        def __init__(self, *args, **kwargs): pass
        def connect(self, *args, **kwargs): pass
        def emit(self, *args, **kwargs): pass
    # --- End Indentation Fix ---


# -------------------------------------------------
# 3 ▸  Simulator class (FP16 Storage / FP32 Compute)
# -------------------------------------------------
class ExplicitWoWSimulator3D_FP16:
    """N-layer WoW sim using FP16 storage, FP32 compute."""
    def __init__(self, *, num_layers=4, grid_size=64,
                 m0=0.1, alpha=2.0, c0=1.0, gamma=1.5,
                 beta=0.7, xi=0.1, dt=0.04, damping=0.001,
                 tension0=5.0, potential_lin0=1.0, potential_cub0=0.2):

        print(f"Init {num_layers}‑layer grid {grid_size}³  (Device: {'GPU' if GPU_ENABLED else 'CPU'})")

        self.N = num_layers
        self.grid_size = grid_size
        self.lock = threading.Lock()

        # Store parameters as Python floats
        self.params = dict(dt=dt, damping=damping, m0=m0, alpha=alpha,
                           c0=c0, gamma=gamma, beta=beta, xi=xi,
                           tension0=tension0, potential_lin0=potential_lin0,
                           potential_cub0=potential_cub0)

        # Precompute layer constants
        self._update_layer_constants()

        # Define dtypes
        self.dtype_store = cp.float16 # Store fields in half-precision
        self.dtype_calc  = cp.float32 # Perform calculations in single-precision

        # Allocate fields on GPU/CPU
        xp = cp # Use alias
        print("Allocating fields...")
        self.phi      = [xp.zeros((grid_size,)*3, dtype=self.dtype_store) for _ in range(self.N)]
        self.phi_old  = [xp.zeros_like(f) for f in self.phi]
        # Pre-allocate scratch buffer for laplacian in calculation precision (FP32)
        self.lap_f32  = xp.empty_like(self.phi[0], dtype=self.dtype_calc)
        print(f"Storage dtype: {self.dtype_store}, Compute dtype: {self.dtype_calc}")

        # 3D Laplacian stencil (in compute precision)
        k_np = np.zeros((3,3,3), dtype=np.float32) # Use float32 for kernel
        k_np[1,1,1] = -6.0
        for dx,dy,dz in [(1,1,0),(1,1,2),(1,0,1),(1,2,1),(0,1,1),(2,1,1)]: k_np[dx,dy,dz]=1.0
        self.kern32 = xp.asarray(k_np, dtype=self.dtype_calc) # Kernel in FP32

        # Simulation time control
        self.update_interval_ms = 33        # ~30 Hz refresh target
        self.t = 0.0
        self.step_count = 0

        self.initialize_field()

    # ---------------- helper ----------------
    def _update_layer_constants(self):
        """Recalculate layer masses and speeds based on params (as floats)."""
        p = self.params
        self.m2 = [(p['m0'] * p['alpha']**n)**2 for n in range(self.N)]
        self.c2 = [(p['c0'] * p['gamma']**n)**2 for n in range(self.N)]
        # print(f"Updated layer constants: m^2={self.m2}, c^2={self.c2}") # Less verbose

    # ---------------- init field ----------------
    def initialize_field(self):
        """Initialize fields with Gaussian pulse in layer 0, noise elsewhere."""
        xp = cp; N = self.grid_size
        print("Initializing φ...")
        with self.lock:
            # Use numpy for coordinate generation on CPU
            x_np, y_np, z_np = [np.arange(N)] * 3
            X_np, Y_np, Z_np = np.meshgrid(x_np, y_np, z_np, indexing='ij')
            cx = cy = cz = N // 2; r2 = max((N / 8.0)**2, 1e-6)
            # Calculate on CPU
            pulse_np = 2.0 * np.exp(-((X_np - cx)**2 + (Y_np - cy)**2 + (Z_np - cz)**2) / (2 * r2))
            noise_np = (np.random.rand(N, N, N) - 0.5) * 0.01

            # Assign to device arrays, casting to storage type (FP16)
            for n in range(self.N):
                data_np = pulse_np if n == 0 else noise_np
                self.phi[n][:]     = xp.asarray(data_np, dtype=self.dtype_store)
                self.phi_old[n][:] = self.phi[n] # Copy on device

            self.t = 0.0; self.step_count = 0
            if GPU_ENABLED: cp.cuda.Stream.null.synchronize() # Ensure GPU init done
        print("Fields Initialized.")

    # ---------------- main step ----------------
    def step(self):
        """Advance all layers by dt using Verlet + coupling (FP32 compute)."""
        xp = cp # Alias
        p = self.params; dt=p['dt']; damp=p['damping']
        beta=p['beta']; xi=p['xi']
        pot_lin=p['potential_lin0']; pot_cub=p['potential_cub0']
        tens=p['tension0']; c0_sq=self.c2[0]

        with self.lock:
            new_phi_f32_list = [None] * self.N # Store FP32 results before casting back

            for n in range(self.N):
                # 1. Fetch fp16 fields -> convert to fp32 for calculation
                phi_f32     = self.phi[n].astype(self.dtype_calc)
                phi_old_f32 = self.phi_old[n].astype(self.dtype_calc)

                # 2. Calculate Laplacian (using fp32 inputs and fp32 kernel)
                #    Store result in the pre-allocated fp32 scratch buffer
                cp_convolve(phi_f32, self.kern32, output=self.lap_f32, mode='wrap')
                # Now self.lap_f32 holds the result

                # 3. Calculate intrinsic acceleration in fp32
                if n == 0:
                    Vp_f32 = -pot_lin * phi_f32 + pot_cub * (phi_f32**3)
                    c2_loc_f32 = xp.asarray(c0_sq / (1.0 + tens * (phi_f32**2) + 1e-6), dtype=self.dtype_calc)
                    accel_intr_f32 = c2_loc_f32 * self.lap_f32 - Vp_f32
                else:
                    c2_n = self.c2[n]; m2_n = self.m2[n] # Python floats
                    accel_intr_f32 = xp.asarray(c2_n * self.lap_f32 - m2_n * phi_f32, dtype=self.dtype_calc)

                # 4. Calculate coupling term in fp32
                coup_f32 = xp.zeros_like(phi_f32, dtype=self.dtype_calc) # FP32 coupling buffer
                xi_eff = xi * (beta**n) # Python float

                if n > 0:
                    # Fetch neighbor, cast to FP32 for calculation
                    phi_nm1_f32 = self.phi[n - 1].astype(self.dtype_calc)
                    coup_f32 += xi_eff * (phi_nm1_f32 - phi_f32)
                if n < self.N - 1:
                    # Fetch neighbor, cast to FP32 for calculation
                    phi_np1_f32 = self.phi[n + 1].astype(self.dtype_calc)
                    coup_f32 += xi_eff * (phi_np1_f32 - phi_f32)

                # 5. Verlet update (all terms are now FP32)
                vel_f32 = phi_f32 - phi_old_f32
                phi_new_f32 = phi_f32 + (1.0 - damp * dt) * vel_f32 \
                              + (dt**2) * (accel_intr_f32 + coup_f32)

                # Store the fp32 result temporarily
                new_phi_f32_list[n] = phi_new_f32

            # 6. Commit results: Cast FP32 back to FP16 storage
            for n in range(self.N):
                self.phi_old[n][:] = self.phi[n] # Store previous FP16 state
                self.phi[n][:]     = new_phi_f32_list[n].astype(self.dtype_store) # Store new state as FP16

            self.t += dt; self.step_count += 1
            if GPU_ENABLED: cp.cuda.Stream.null.synchronize()

    # ---------------- helpers ----------------
    def get_field_copy(self, layer=0) -> np.ndarray:
        """Return a thread-safe *NumPy* copy (FP16 or FP32 depending on storage)."""
        if not (0 <= layer < self.N):
             print(f"Error: Invalid layer {layer}"); return None
        with self.lock:
            field_device = self.phi[layer]
            # Return in compute precision (float32) for visualization consistency
            field_device_f32 = field_device.astype(self.dtype_calc)
            return cp.asnumpy(field_device_f32) if GPU_ENABLED else np.copy(field_device_f32)

    def update_params(self, d):
        """Update parameters (as standard Python floats)."""
        with self.lock:
            params_changed = False
            for k, v in d.items():
                if k in self.params:
                    try:
                        new_val = float(v)
                        if self.params[k] != new_val: self.params[k] = new_val; params_changed = True
                    except (ValueError, TypeError): pass
            if params_changed and any(k in d for k in ('m0','alpha','c0','gamma')):
                self._update_layer_constants()

    def reset_simulation(self):
        """Reset simulation state."""
        with self.lock: self.initialize_field()

# -------------------------------------------------
# 4 ▸  Simulation thread (Unchanged)
# -------------------------------------------------
class SimulationWorker(QObject):
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str) # Add error signal
    def __init__(self, sim): super().__init__(); self.sim = sim; self._running = False
    def run(self):
        self._running = True; print("Sim thread started.")
        while self._running:
            t0 = time.perf_counter()
            try: self.sim.step()
            except Exception as e_sim:
                 err_msg = f"!!! ERROR in sim step: {e_sim}"
                 print(err_msg); traceback.print_exc()
                 self.error_occurred.emit(err_msg); self._running = False; break
            elapsed = (time.perf_counter() - t0) * 1000
            sleep_ms = max(1, self.sim.update_interval_ms - elapsed)
            time.sleep(sleep_ms / 1000.0)
        print("Sim thread finished."); self.finished.emit()
    def stop(self): print("Stopping sim thread..."); self._running = False


# -------------------------------------------------
# 5 ▸  Qt MainWindow (Updated for FP16 sim, robust drawing)
# -------------------------------------------------
class MainWindow(QMainWindow):
    """GUI for Explicit WoW Simulation (GPU FP16 version)."""
    def __init__(self, sim):
        super().__init__()
        # Check if GUI is supported first
        if not PYVISTA_AVAILABLE:
            print("GUI cannot start because required libraries are missing.")
            sys.exit("Exiting due to missing GUI libraries.")

        self.sim = sim # Expects ExplicitWoWSimulator3D_FP16
        self.setWindowTitle("Interactive 3D WoW Simulation (GPU FP16)")
        self.resize(1300, 800)

        # --- GUI State ---
        self.vis_layer_index = 0
        self.iso_value = 1.0
        self.pv_grid = None # Initialize grid attribute

        # --- Layout Setup ---
        cw = QWidget(); self.setCentralWidget(cw)
        main_layout = QHBoxLayout(cw)
        pf = QFrame(); pf.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        pv_layout = QVBoxLayout(pf)
        try: self.plotter = QtInteractor(pf)
        except Exception as e_plotter: print(f"FATAL ERROR initializing PyVista: {e_plotter}"); sys.exit(1)
        pv_layout.addWidget(self.plotter); main_layout.addWidget(pf, stretch=3)
        ctrl = QWidget(); ctrl.setFixedWidth(380); cl = QVBoxLayout(ctrl); main_layout.addWidget(ctrl, stretch=1)

        # --- Setup Controls & Scene ---
        self._setup_controls(cl)
        self._scene_setup()

        # --- Worker & Timer ---
        self.worker = None; self.thread = None
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._refresh)
        self.update_timer.setInterval(self.sim.update_interval_ms)

    # --- Control Setup & Callbacks ---
    def _slider_conf(self, n):
        # (Same as before)
        if n=='dt': return 0.01,0.1,1000,3
        if n=='damping': return 0,0.01,10000,4
        if n=='m0': return 0.01, 5.0, 100, 2
        if n=='alpha': return 1.0, 4.0, 100, 2
        if n=='c0': return 0.1, 5.0, 100, 2
        if n=='gamma': return 1.0, 4.0, 100, 2
        if n=='beta': return 0.1, 1.0, 100, 2
        if n=='xi': return 0.0, 1.0, 100, 2
        if n=='tension0': return 0, 20, 10, 1
        if n=='potential_lin0': return 0.1, 2, 100, 2
        if n=='potential_cub0': return 0, 1, 100, 2
        return 0,1,100,2

    def _setup_controls(self, cl):
        # (Same setup using self.sim.params as before)
        hl = QHBoxLayout(); self.start_b = QPushButton("Start"); self.start_b.clicked.connect(self.start)
        self.stop_b = QPushButton("Stop"); self.stop_b.clicked.connect(self.stop); self.stop_b.setEnabled(False)
        self.reset_b = QPushButton("Reset"); self.reset_b.clicked.connect(self.reset)
        hl.addWidget(self.start_b); hl.addWidget(self.stop_b); hl.addWidget(self.reset_b); cl.addLayout(hl)
        cl.addWidget(QLabel("Simulation Parameters:"))
        self.param_controls = {}
        for name in self.sim.params:
             h = QHBoxLayout(); lbl_name = QLabel(name.replace('_', ' ').title()); lbl_name.setMinimumWidth(110)
             use_spinbox = name in ['dt', 'damping', 'beta', 'xi']
             mn, mx, sc, pr = self._slider_conf(name)
             if use_spinbox:
                 widget = QDoubleSpinBox(); widget.setRange(mn, mx); widget.setDecimals(pr + 1); widget.setSingleStep(10**(-pr))
                 widget.setValue(self.sim.params[name]); widget.valueChanged.connect(lambda v, n=name, p=pr: self._param_changed(n, v, p))
             else:
                 widget = QSlider(Qt.Horizontal); widget.setRange(int(mn * sc), int(mx * sc)); widget.setValue(int(self.sim.params[name] * sc))
                 widget.valueChanged.connect(lambda v, n=name, s=sc, p=pr: self._param_changed(n, v / s, p))
             val_lbl = QLabel(f"{self.sim.params[name]:.{pr}f}"); val_lbl.setMinimumWidth(60)
             h.addWidget(lbl_name); h.addWidget(widget); h.addWidget(val_lbl); cl.addLayout(h)
             self.param_controls[name] = {'w': widget, 'l': val_lbl, 'pr': pr}
             if not use_spinbox: self.param_controls[name]['sc'] = sc
        vis_group = QGroupBox("Visualization"); gl = QVBoxLayout(vis_group)
        layer_sel_layout = QHBoxLayout(); layer_sel_layout.addWidget(QLabel("Visualize Layer:"))
        self.vis_layer_buttons = []
        for i in range(self.sim.N):
            rb = QRadioButton(f"{i}"); rb.setChecked(i == self.vis_layer_index)
            rb.toggled.connect(lambda checked, idx=i: self._vis_layer_changed(idx, checked))
            layer_sel_layout.addWidget(rb); self.vis_layer_buttons.append(rb)
        layer_sel_layout.addStretch(); gl.addLayout(layer_sel_layout)
        iso_layout = QHBoxLayout(); iso_label = QLabel("Iso Value:")
        self.iso_slider = QSlider(Qt.Horizontal); self.iso_slider.setRange(-300, 300); self.iso_slider.setValue(int(self.iso_value * 100))
        self.iso_value_label = QLabel(f"{self.iso_value:.2f}"); self.iso_value_label.setMinimumWidth(40)
        self.iso_slider.valueChanged.connect(lambda v: self._iso_changed(v / 100.0))
        iso_layout.addWidget(iso_label); iso_layout.addWidget(self.iso_slider); iso_layout.addWidget(self.iso_value_label)
        gl.addLayout(iso_layout); cl.addWidget(vis_group); cl.addStretch()

    def _param_changed(self, name, value, precision):
        self.sim.update_params({name: value})
        if name in self.param_controls: self.param_controls[name]['l'].setText(f"{value:.{precision}f}")

    def _vis_layer_changed(self, index, checked):
        if checked: self.vis_layer_index = index; print(f"Switched vis to Layer {index}"); self._draw_once()

    def _iso_changed(self, value):
        self.iso_value = value; self.iso_value_label.setText(f"{value:.2f}"); self._draw_once()

    # --- Visualization ---
    def _scene_setup(self):
        """Setup initial PyVista scene."""
        print("Setting up PyVista scene...")
        self.plotter.clear_actors()
        self.plotter.add_axes()
        phi_data_np = self.sim.get_field_copy(self.vis_layer_index)
        if phi_data_np is None or phi_data_np.size == 0: print("Error: No init field data."); return
        self.pv_grid = pv.ImageData()
        self.pv_grid.dimensions = np.array(phi_data_np.shape) + 1
        self.pv_grid.origin = (0,0,0); self.pv_grid.spacing = (1,1,1)
        self.plotter.add_mesh(self.pv_grid.outline(), color='grey', name='outline')
        self._draw_once(); self.plotter.camera_position = 'xy'; self.plotter.reset_camera()
        print("PyVista scene setup complete.")

    def _draw_once(self):
        """Fetch data, update grid, and redraw isosurface. Returns True on success."""
        phi_n_np = self.sim.get_field_copy(self.vis_layer_index)
        if phi_n_np is None or phi_n_np.size == 0: return False

        try:
            target_dims = tuple(map(int, np.array(phi_n_np.shape) + 1))
            if not hasattr(self, 'pv_grid') or self.pv_grid.dimensions != target_dims:
                # print("Recreating PyVista grid...") # Reduce console noise
                self.pv_grid = pv.ImageData(dimensions=target_dims, origin=(0,0,0), spacing=(1,1,1))
                if 'outline' not in self.plotter.actors: self.plotter.add_mesh(self.pv_grid.outline(), color='grey', name='outline')

            if self.pv_grid.n_cells != phi_n_np.size:
                 print(f"Grid/Data size mismatch!"); return False

            # Assign cell data (needs numpy array on CPU)
            self.pv_grid.cell_data['phi'] = phi_n_np.ravel(order='F')
            # Convert to point data for contouring
            grid_pdata = self.pv_grid.cell_data_to_point_data()
            if 'phi' not in grid_pdata.point_data: return False

            actor_name = f'layer_{self.vis_layer_index}_iso'
            self.plotter.remove_actor(actor_name, render=False) # Remove old actor

            # Create new isosurface if iso_value is within data range
            iso_val = self.iso_value
            min_phi, max_phi = grid_pdata.get_data_range('phi')
            new_isosurface = None
            if min_phi <= iso_val <= max_phi:
                new_isosurface = grid_pdata.contour([iso_val], scalars='phi', preference='points', compute_normals=True)

            # Add mesh only if it's valid
            if new_isosurface is not None and new_isosurface.n_points > 0 and hasattr(new_isosurface, 'n_faces') and new_isosurface.n_faces > 0:
                self.plotter.add_mesh(new_isosurface, name=actor_name, scalars='phi', cmap='coolwarm',
                                      opacity=0.7, show_scalar_bar=False) # Maybe turn bar off
            # else: Optional: print(f"No valid surface at iso={iso_val:.2f}")

            return True # Success

        except Exception as e_draw:
            print(f"Error during drawing layer {self.vis_layer_index}: {e_draw}")
            traceback.print_exc()
            return False

    def _refresh(self):
        """Timer callback to refresh visualization."""
        if hasattr(self, 'worker') and self.worker and hasattr(self.worker, '_running') and self.worker._running:
            success = self._draw_once()
            # Render only on successful draw to avoid error loops trying to render invalid states
            if success:
                 self.plotter.render()

    # --- Simulation Control Methods ---
    def start(self):
        if not (hasattr(self, 'worker') and self.worker and hasattr(self.worker, '_running') and self.worker._running):
             for name, ctrl_info in self.param_controls.items():
                 widget = ctrl_info['w']
                 if isinstance(widget, QSlider): value = widget.value() / ctrl_info.get('sc', 1)
                 else: value = widget.value()
                 self.sim.update_params({name: value})
             self.worker = SimulationWorker(self.sim); self.thread = QThread()
             self.worker.moveToThread(self.thread); self.thread.started.connect(self.worker.run)
             self.worker.finished.connect(self.thread.quit); self.worker.finished.connect(self.worker.deleteLater)
             self.thread.finished.connect(self.thread.deleteLater); self.thread.start()
             self.update_timer.start()
             self.start_b.setEnabled(False); self.stop_b.setEnabled(True); self.reset_b.setEnabled(False)
             print("Simulation started via GUI.")

    def stop(self):
        if hasattr(self, 'worker') and self.worker: self.worker.stop()
        self.update_timer.stop()
        if hasattr(self, 'thread') and self.thread and self.thread.isRunning():
            self.thread.quit(); self.thread.wait(500) # Shorter wait
        self.worker = None; self.thread = None
        self.start_b.setEnabled(True); self.stop_b.setEnabled(False); self.reset_b.setEnabled(True)
        print("Simulation stopped via GUI.")

    def reset(self):
        was_running = (hasattr(self, 'worker') and self.worker and hasattr(self.worker, '_running') and self.worker._running)
        if was_running: self.stop()
        for name, ctrl_info in self.param_controls.items():
             widget = ctrl_info['w']
             if isinstance(widget, QSlider): value = widget.value() / ctrl_info.get('sc', 1)
             else: value = widget.value()
             self.sim.update_params({name: value})
        self.sim.reset_simulation()
        self._draw_once(); self.plotter.render()
        print("Simulation reset.")

    def closeEvent(self, ev):
        print("Closing window...")
        self.stop()
        if hasattr(self, 'plotter') and self.plotter:
             try: self.plotter.close()
             except: pass
        super().closeEvent(ev)

# -------------------------------------------------
# 6 ▸  main execution
# -------------------------------------------------
if __name__ == '__main__':
    app = QApplication.instance() # Check if app already exists
    if app is None: app = QApplication(sys.argv)

    if not PYVISTA_AVAILABLE:
        print("\nCannot run GUI without required libraries (PyQt5, PyVista, PyVistaQT).")
        sys.exit(1)

    # Initialize simulator (adjust grid size based on GPU memory)
    try:
        # Attempt to use a larger grid size, e.g., 64^3 or 128^3 if sufficient VRAM
        grid_size_to_try = 320 # Start with 64 as default for GPU
        print(f"Attempting to initialize simulator with grid_size={grid_size_to_try}...")
        sim_instance = ExplicitWoWSimulator3D_FP16(num_layers=4, grid_size=grid_size_to_try)
    except Exception as e_mem: # Catch potential memory errors on GPU
        print(f"\nWarning: Failed to init with grid_size={grid_size_to_try} ({e_mem})")
        grid_size_to_try = 32 # Fallback to smaller size
        print(f"Falling back to grid_size={grid_size_to_try}")
        try:
            sim_instance = ExplicitWoWSimulator3D_FP16(num_layers=4, grid_size=grid_size_to_try)
        except Exception as e_mem2:
             print(f"\nFATAL ERROR: Failed to init even with grid_size={grid_size_to_try} ({e_mem2})")
             print("Check available GPU memory or reduce grid size further.")
             sys.exit(1)

    # Create and show GUI window
    window = MainWindow(sim_instance)
    window.show()
    print("Starting Qt event loop...")
    exit_code = app.exec_()
    print(f"Qt event loop finished (exit code {exit_code}).")
    sys.exit(exit_code)
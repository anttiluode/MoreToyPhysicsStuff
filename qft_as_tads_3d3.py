# -*- coding: utf-8 -*-
import numpy as np
from scipy.ndimage import convolve # Use ndimage for 3D convolution
import time
from pathlib import Path # For creating directories and handling paths

# Note: No visualization or GUI imports here - needs separate implementation

class EmergentParticleSimulator3D:
    """
    Conceptual sketch for a 3D TADS/WoW Clockfield Simulator.
    Focuses on core physics, includes data saving, omits visualization.
    """
    def __init__(self, grid_size=32): # START SMALL (e.g., 32 or 64)
        self.grid_size = grid_size
        print(f"Initializing 3D grid: {grid_size}x{grid_size}x{grid_size} = {grid_size**3} points")

        # --- Parameters (Set directly, no GUI) ---
        self.dt = 0.04 # Often needs smaller dt for stability in 3D/higher derivatives
        self.damping = 0.001
        self.base_c_sq = 1.0
        self.tension_factor = 5.0 # TADS: Intensity-dependent speed
        self.potential_lin = 1.0 # TADS: Potential part
        self.potential_cub = 0.2 # TADS: Potential part
        self.biharmonic_gamma = 0.02 # WoW: Implicit coupling term strength

        # --- Internal State ---
        self.phi = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
        self.phi_old = np.zeros_like(self.phi)
        self.t = 0.0
        self.step_count = 0

        # --- 3D Laplacian Kernel (7-point stencil) ---
        self.laplacian_kernel = np.zeros((3, 3, 3), dtype=np.float64)
        self.laplacian_kernel[1, 1, 1] = -6.0
        self.laplacian_kernel[1, 1, 0] = 1.0
        self.laplacian_kernel[1, 1, 2] = 1.0
        self.laplacian_kernel[1, 0, 1] = 1.0
        self.laplacian_kernel[1, 2, 1] = 1.0
        self.laplacian_kernel[0, 1, 1] = 1.0
        self.laplacian_kernel[2, 1, 1] = 1.0

        # --- Initialize ---
        self.initialize_field('gaussian_pulse_3d') #'random'

    def initialize_field(self, mode='gaussian_pulse_3d'):
        """Initialize the 3D field configuration."""
        print(f"Initializing field with mode: {mode}")
        if mode == 'gaussian_pulse_3d':
            x = np.arange(self.grid_size)
            y = np.arange(self.grid_size)
            z = np.arange(self.grid_size)
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            cx = cy = cz = self.grid_size // 2
            radius = self.grid_size / 8.0
            radius_sq = max(radius**2, 1e-6)
            self.phi[:] = 2.0 * np.exp(-((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / (2 * radius_sq))
        elif mode == 'random':
             np.random.seed(int(time.time()))
             # Smaller amplitude random noise
             self.phi = (np.random.rand(self.grid_size, self.grid_size, self.grid_size) - 0.5) * 0.1
        else: # Default to zeros
            self.phi = np.zeros((self.grid_size, self.grid_size, self.grid_size), dtype=np.float64)

        self.phi_old = np.copy(self.phi)
        self.t = 0.0
        self.step_count = 0
        print("3D Field Initialized.")

    def _laplacian(self, field):
        return convolve(field, self.laplacian_kernel, mode='wrap')

    def _biharmonic(self, field):
        lap_field = self._laplacian(field)
        return self._laplacian(lap_field)

    def _potential_deriv(self, phi):
        return (-self.potential_lin * phi + self.potential_cub * (phi**3))

    def _local_speed_sq(self, phi):
         intensity = phi**2
         return self.base_c_sq / (1.0 + self.tension_factor * intensity + 1e-9)

    def step(self):
        """Perform one time step of the 3D simulation."""
        lap_phi = self._laplacian(self.phi)
        biharm_phi = self._biharmonic(self.phi)
        c2 = self._local_speed_sq(self.phi)
        V_prime = self._potential_deriv(self.phi)
        acceleration = (c2 * lap_phi) - V_prime - (self.biharmonic_gamma * biharm_phi)
        velocity = self.phi - self.phi_old
        phi_new = self.phi + (1.0 - self.damping*self.dt)*velocity + (self.dt**2)*acceleration
        self.phi_old = self.phi
        self.phi = phi_new
        self.t += self.dt
        self.step_count += 1

# --- Example Script Usage (Command Line Execution) ---
if __name__ == "__main__":
    print("Starting 3D Simulation Example with Data Saving...")
    start_time = time.time()

    # --- Simulation Parameters ---
    GRID_SIZE = 128    # Keep this small initially!
    TOTAL_STEPS = 250 # Number of steps to run

    # --- Data Saving Parameters ---
    SAVE_INTERVAL = 25  # Save data every N steps
    OUTPUT_DIR = Path("./phi_data_3d") # Directory to save .npy files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Create dir if it doesn't exist
    print(f"Data will be saved every {SAVE_INTERVAL} steps to: {OUTPUT_DIR.resolve()}")

    # --- Initialize and Run ---
    sim = EmergentParticleSimulator3D(grid_size=GRID_SIZE)
    # Set initial condition if desired (e.g., 'random')
    # sim.initialize_field('random')

    print(f"Running {TOTAL_STEPS} steps on a {GRID_SIZE}^3 grid...")
    for i in range(TOTAL_STEPS):
        sim.step()

        # --- Check for Saving Data ---
        save_now = (i + 1) % SAVE_INTERVAL == 0 # Check interval
        save_now = save_now or (i == 0)         # Always save step 1
        save_now = save_now or (i == TOTAL_STEPS - 1) # Always save last step

        if save_now:
            max_phi_val = np.max(sim.phi)
            filename = OUTPUT_DIR / f"phi_step_{i+1:06d}.npy" # Pad step number for sorting
            np.save(filename, sim.phi)
            # Print status update less frequently or only when saving
            print(f"  Step {i+1:6d}/{TOTAL_STEPS}, Sim Time: {sim.t:8.3f}, Max Phi: {max_phi_val:8.4f} -> Saved: {filename.name}")
        elif (i + 1) % 50 == 0: # Print basic progress more often
             max_phi_val = np.max(sim.phi)
             print(f"  Step {i+1:6d}/{TOTAL_STEPS}, Sim Time: {sim.t:8.3f}, Max Phi: {max_phi_val:8.4f}")


    # --- End Simulation ---
    end_time = time.time()
    print(f"\nFinished {TOTAL_STEPS} steps.")
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
    print(f"Final Simulation Time: {sim.t:.3f}")
    print(f"Final Max Phi Value: {np.max(sim.phi):.4f}")
    print(f"Final Min Phi Value: {np.min(sim.phi):.4f}")
    print(f"Data saved in: {OUTPUT_DIR.resolve()}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import Axes3D if using older Matplotlib, usually not needed now
# from mpl_toolkits.mplot3d import Axes3D
import ast # To safely evaluate the string representation of lists/tuples
from pathlib import Path
import sys

# --- Configuration ---
PARTICLE_LOG_FILE = Path("./particle_log.csv")
# Plotting parameters
MARKER_SIZE = 2       # Size of points in the scatter plot
MARKER_ALPHA = 0.1     # Transparency of points (lower for dense plots)
COLOR_BY = 'time'    # Color points by 'time' or 'amplitude' or None
CMAP = 'viridis'      # Colormap to use if coloring points

def parse_particle_string(particle_str):
    """Safely parse the string representation of the particle list."""
    try:
        # Replace nan with a placeholder string if necessary, though ast should handle Python's nan/inf
        # particle_str = particle_str.replace('nan', '"NAN_PLACEHOLDER"')
        particles = ast.literal_eval(particle_str)
        # Ensure it's a list of tuples/lists with 3 elements (x, y, amp)
        if isinstance(particles, list):
            # Filter out any potential placeholders if used, or invalid entries
            # cleaned_particles = [p for p in particles if isinstance(p, (tuple, list)) and len(p) == 3 and "NAN" not in str(p)]
            # For now, assume valid list of tuples/lists
             cleaned_particles = [p for p in particles if isinstance(p, (tuple, list)) and len(p) == 3]
             return cleaned_particles
        else:
            return [] # Return empty list if parsing didn't yield a list
    except (ValueError, SyntaxError, TypeError) as e:
        # Handle potential errors if the string is not a valid list representation
        # print(f"Warning: Could not parse particle string: {particle_str[:100]}... Error: {e}")
        return [] # Return empty list on error

if __name__ == "__main__":
    print(f"Attempting to load 2D particle log: {PARTICLE_LOG_FILE}")

    if not PARTICLE_LOG_FILE.is_file():
        print(f"Error: File not found: {PARTICLE_LOG_FILE.resolve()}")
        print("Please make sure 'particle_log.csv' from the 2D simulation is in the same directory.")
        sys.exit(1)

    # --- Load Data ---
    try:
        df = pd.read_csv(PARTICLE_LOG_FILE)
        print(f"Loaded {len(df)} time steps.")
        if 'particles' not in df.columns or 'time' not in df.columns:
             print("Error: CSV must contain 'time' and 'particles' columns.")
             sys.exit(1)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)

    # --- Extract Particle Data ---
    print("Extracting particle coordinates and times...")
    all_times = []
    all_x = []
    all_y = []
    all_amps = []

    parse_errors = 0
    for index, row in df.iterrows():
        time_step = row['time']
        # Use the safe parser function
        particles = parse_particle_string(row['particles'])
        if particles is None: # Check if parser returned None due to error
            parse_errors += 1
            continue # Skip this row if parsing failed

        for p_data in particles:
             # Simple check if data looks like numbers, skip otherwise
             try:
                 x, y, amp = map(float, p_data) # Convert to float
                 all_times.append(time_step)
                 all_x.append(x)
                 all_y.append(y)
                 all_amps.append(amp)
             except (ValueError, TypeError):
                 # Skip particle data that isn't convertible to float
                 parse_errors += 1
                 continue

    if parse_errors > 0:
         print(f"Warning: Skipped {parse_errors} entries due to parsing issues.")

    if not all_times:
        print("Error: No valid particle data extracted. Cannot plot.")
        sys.exit(1)

    print(f"Extracted {len(all_times)} total particle positions across time.")

    # --- Create 3D Scatter Plot ---
    print("Generating 3D spacetime plot...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Determine colors
    colors = None
    if COLOR_BY == 'time':
        colors = all_times
        cbar_label = 'Time'
    elif COLOR_BY == 'amplitude':
        colors = all_amps
        cbar_label = 'Amplitude (Phi)'
    else:
        cbar_label = None


    # Create scatter plot
    try:
        if colors is not None:
            sc = ax.scatter(all_x, all_y, all_times,
                            s=MARKER_SIZE, c=colors, cmap=CMAP, alpha=MARKER_ALPHA, depthshade=True)
            # Add color bar
            cbar = fig.colorbar(sc, shrink=0.6)
            cbar.set_label(cbar_label)
        else:
             sc = ax.scatter(all_x, all_y, all_times,
                             s=MARKER_SIZE, alpha=MARKER_ALPHA, depthshade=True)

        # --- Setup Plot ---
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Time')
        ax.set_title('Particle Positions in Spacetime (X vs Y vs Time)')
        ax.grid(True)

        # Optional: Set limits if needed, e.g., based on grid size
        # grid_info = # Need grid size from sim, assume 128 if not available
        # ax.set_xlim(0, 128)
        # ax.set_ylim(0, 128)

        print("Displaying plot...")
        plt.show()
        print("Plot window closed.")

    except Exception as e:
         print(f"An error occurred during plotting: {e}")
         import traceback
         traceback.print_exc()

    print("Script finished.")
# -*- coding: utf-8 -*-
import numpy as np
import pyvista as pv
from pathlib import Path
import re
import sys
import traceback # Import for better error reporting

def find_latest_npy_file(data_dir):
    """Finds the .npy file with the highest step number in the directory."""
    print(f"Searching for '*.npy' files in: {data_dir.resolve()}")
    npy_files = list(data_dir.glob("phi_step_*.npy"))
    if not npy_files: return None
    latest_file = None; highest_step = -1
    pattern = re.compile(r"phi_step_(\d+)\.npy")
    for f in npy_files:
        match = pattern.search(f.name)
        if match:
            try:
                step = int(match.group(1))
                if step > highest_step: highest_step = step; latest_file = f
            except ValueError: continue
    if latest_file: print(f"  Using file: {latest_file.name}")
    else: print("  Could not identify a latest file.")
    return latest_file

if __name__ == "__main__":
    # --- Configuration ---
    DEFAULT_DATA_DIR = Path("./phi_data_3d")
    VISUALIZATION_TYPE = 'isosurface' # Defaulting to isosurface
    POS_ISO_VALUE = 2 # Adjust based on data range!
    NEG_ISO_VALUE = -2 # Adjust based on data range!
    OPACITY_MAP = 'sigmoid' # For volume rendering if used

    # --- Get Data Directory ---
    data_dir = DEFAULT_DATA_DIR
    if not data_dir.is_dir():
        print(f"Error: Directory not found: {data_dir.resolve()}")
        sys.exit(1)

    # --- Find the latest file automatically ---
    file_to_load = find_latest_npy_file(data_dir)
    if file_to_load is None:
        print("No suitable '.npy' file found to load. Exiting.")
        sys.exit(1)

    # --- Load Data ---
    try:
        print(f"Loading data from {file_to_load} for PyVista...")
        phi_data = np.load(file_to_load)
        print(f"Data loaded successfully. Shape: {phi_data.shape}")
        if phi_data.ndim != 3:
            raise ValueError(f"Expected 3D data array, but got shape {phi_data.shape}")
    except Exception as e:
        print(f"Error loading file {file_to_load}: {e}")
        sys.exit(1)

    # --- Create PyVista Grid using ImageData ---
    try:
        grid = pv.ImageData()
        # Dimensions are number of points (N+1 for N cells)
        grid.dimensions = np.array(phi_data.shape) + 1
        # Origin and spacing can be set if needed, default is (0,0,0) and (1,1,1)
        # grid.origin = (0, 0, 0)
        # grid.spacing = (1, 1, 1)

        # *** Assign data to CELLS ***
        if np.prod(phi_data.shape) != grid.n_cells:
             raise ValueError(f"Data size {phi_data.size} does not match number of grid cells {grid.n_cells}")
        grid.cell_data['phi'] = phi_data.ravel(order='F') # Ravel Fortran style for cell data
        print(f"Created grid. Dimensions: {grid.dimensions}, Cells: {grid.n_cells}")
        print(f"Assigned cell data 'phi'.")
        print(f"Data range: Min={grid.cell_data['phi'].min():.4f}, Max={grid.cell_data['phi'].max():.4f}")

        # *** Convert cell data to point data if needed for visualization ***
        # Many PyVista filters (like contour) operate on point data
        print("Converting cell data to point data...")
        grid_pdata = grid.cell_data_to_point_data()
        print("Conversion done. Point data 'phi' now available on grid_pdata.")
        # Verify point data exists
        if 'phi' not in grid_pdata.point_data:
             raise RuntimeError("Failed to create point data 'phi' after conversion.")
        print(f"Point data range: Min={grid_pdata.point_data['phi'].min():.4f}, Max={grid_pdata.point_data['phi'].max():.4f}")


    except Exception as e_general:
         print(f"\nAn unexpected error occurred creating the PyVista grid: {e_general}")
         traceback.print_exc()
         sys.exit(1)

    # --- Set up Plotter ---
    plotter = pv.Plotter(window_size=[800, 800])
    plotter.add_axes()

    # --- Add Visualization ---

    if VISUALIZATION_TYPE == 'volume':
        # Volume rendering can often use cell data directly
        print(f"Adding volume rendering (using cell data). Opacity map: {OPACITY_MAP}")
        plotter.add_volume(grid, scalars='phi', cmap='viridis', opacity=OPACITY_MAP)

    elif VISUALIZATION_TYPE == 'isosurface':
        print(f"Adding isosurfaces at phi = {POS_ISO_VALUE} and {NEG_ISO_VALUE}")
        # Use the grid with POINT data for contouring
        try:
            # Contour using the grid_pdata object
            pos_contours = grid_pdata.contour(isosurfaces=[POS_ISO_VALUE], scalars='phi', compute_normals=True)
            if pos_contours.n_points > 0:
                plotter.add_mesh(pos_contours, color='yellow', opacity=0.7, name='pos_iso')
                print(f"  Added positive isosurface ({pos_contours.n_points} points).")
            else: print("  No points generated for positive isosurface.")
        except Exception as e_pos_iso: print(f"  Error generating positive isosurface: {e_pos_iso}")

        try:
            neg_contours = grid_pdata.contour(isosurfaces=[NEG_ISO_VALUE], scalars='phi', compute_normals=True)
            if neg_contours.n_points > 0:
                plotter.add_mesh(neg_contours, color='purple', opacity=0.7, name='neg_iso')
                print(f"  Added negative isosurface ({neg_contours.n_points} points).")
            else: print("  No points generated for negative isosurface.")
        except Exception as e_neg_iso: print(f"  Error generating negative isosurface: {e_neg_iso}")

        # Add outline for context
        plotter.add_mesh(grid.outline(), color='grey')


    elif VISUALIZATION_TYPE == 'slices':
        print("Adding orthogonal slices (using point data).")
        # Slicing usually works better with point data too
        plotter.add_mesh_slice_orthogonal(grid_pdata, scalars='phi', cmap='viridis')

    else:
         print(f"Unknown visualization type: {VISUALIZATION_TYPE}. Adding outline.")
         plotter.add_mesh(grid.outline(), color='k')


    # --- Show Plot ---
    print("Launching PyVista interactive plot window...")
    try:
        plotter.show()
        print("PyVista plot window closed.")
    except Exception as e_show:
        print(f"An error occurred displaying the PyVista plot: {e_show}")
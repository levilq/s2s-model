# src/preprocessing/load_dem.py (from my last response)

import os
import time
import numpy as np
import rasterio
from rasterio.fill import fillnodata
from landlab import RasterModelGrid
import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_dem(dem_path, mask_path=None):
    """Load DEM data and optionally a watershed mask from .tif files."""
    start_time = time.time()
    with rasterio.open(dem_path) as src:
        elevation = src.read(1)
        nodata = src.nodata
        bounds = src.bounds
        res = src.res
        # Pre-fill depressions in the DEM
        if nodata is not None:
            mask = elevation != nodata
            elevation_filled = fillnodata(elevation, mask=mask, max_search_distance=10)
        else:
            elevation_filled = elevation.copy()
    # Load watershed mask if provided
    if mask_path:
        with rasterio.open(mask_path) as mask_src:
            watershed_mask = mask_src.read(1)
            # Ensure mask is binary (1 = inside watershed, 0 = outside)
            watershed_mask = (watershed_mask == 1).astype(bool)
    else:
        watershed_mask = None
    end_time = time.time()
    print(f"Time to load DEM: {end_time - start_time:.2f} seconds")
    return elevation_filled, bounds, res, nodata, watershed_mask


def create_landlab_grid(elevation, dx, dy, nodata, watershed_mask=None):
    """Create a Landlab grid from DEM data with two outlets (seepage and overflow)."""
    start_time = time.time()
    rows, cols = elevation.shape
    grid = RasterModelGrid((rows, cols), xy_spacing=(dx, dy))

    # Convert to float64 and handle no-data values
    elevation_float = elevation.astype(np.float64)
    if nodata is not None:
        nodata_mask = elevation == nodata
        elevation_float[nodata_mask] = 0.0  # Set no-data to 0 instead of NaN
    else:
        nodata_mask = np.zeros_like(elevation, dtype=bool)

    # Use watershed mask if provided
    if watershed_mask is not None:
        nodata_mask = ~watershed_mask

    # Debug: Check the extent of the no-data mask
    print(f"Number of no-data cells: {np.sum(nodata_mask)} out of {rows * cols} total cells")

    # Add elevation field
    grid.add_field('topographic__elevation', elevation_float, at='node', units='m')

    # Set no-data nodes to closed
    grid.set_nodata_nodes_to_closed(grid.at_node['topographic__elevation'], 0.0)

    # Set valid nodes (inside the watershed) to core nodes
    flat_nodata_mask = nodata_mask.ravel()
    grid.status_at_node[flat_nodata_mask] = grid.BC_NODE_IS_CLOSED
    grid.status_at_node[~flat_nodata_mask] = grid.BC_NODE_IS_CORE

    # Ensure perimeter nodes of the entire grid are closed
    grid.set_closed_boundaries_at_grid_edges(True, True, True, True)

    # Define outlet nodes
    seepage_outlet_node = 60 * cols + 149  # Just before the dam
    overflow_outlet_node = 60 * cols + 150  # On the dam crest

    if flat_nodata_mask[seepage_outlet_node]:
        print("Warning: Seepage outlet node is in a no-data area.")
    else:
        grid.status_at_node[seepage_outlet_node] = grid.BC_NODE_IS_FIXED_VALUE

    if flat_nodata_mask[overflow_outlet_node]:
        print("Warning: Overflow outlet node is in a no-data area.")
    else:
        grid.status_at_node[overflow_outlet_node] = grid.BC_NODE_IS_CLOSED  # Initially closed

    # Debug: Print boundary conditions
    print(f"Number of closed nodes: {np.sum(grid.status_at_node == grid.BC_NODE_IS_CLOSED)}")
    print(f"Number of core nodes: {np.sum(grid.status_at_node == grid.BC_NODE_IS_CORE)}")
    print(f"Number of fixed value (open) nodes: {np.sum(grid.status_at_node == grid.BC_NODE_IS_FIXED_VALUE)}")

    end_time = time.time()
    print(f"Time to create grid: {end_time - start_time:.2f} seconds")
    return grid, nodata_mask


def plot_dem(grid):
    """Display the DEM plot without saving."""
    print("Starting to plot DEM...")
    start_time = time.time()
    plt.figure(figsize=(10, 8))
    plt.imshow(grid.at_node['topographic__elevation'].reshape(grid.shape), cmap='terrain')
    plt.colorbar(label='Elevation (m)')
    plt.title('Sarez Lake DEM')
    plt.show()
    end_time = time.time()
    print(f"Time to display DEM: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    dem_path = os.path.join(DATA_DIR, 'sarez1000m.tif')
    mask_path = os.path.join(DATA_DIR, 'sarez_watershed_mask.tif') if os.path.exists(
        os.path.join(DATA_DIR, 'sarez_watershed_mask.tif')) else None
    file_size = os.path.getsize(dem_path) / (1024 * 1024)
    print(f"DEM file size: {file_size:.2f} MB")

    elevation, bounds, (dx, dy), nodata, watershed_mask = load_dem(dem_path, mask_path)
    print(f"DEM loaded. Shape: {elevation.shape}, Resolution: ({dx}, {dy}), No-data value: {nodata}")

    grid, nodata_mask = create_landlab_grid(elevation, dx, dy, nodata, watershed_mask)
    print(f"Landlab grid created with {grid.number_of_nodes} nodes.")

    plot_dem(grid)
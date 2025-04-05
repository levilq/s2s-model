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

    # Ensure perimeter nodes of the entire grid are closed
    grid.set_watershed_boundary_condition(grid.field_values('topographic__elevation'), nodata_value=0.0)
    
    #TODO: When the model is running well, add one more outlet for the seepage and close the other one until the lake level reaches the dam crest.

    end_time = time.time()
    print(f"Time to create grid: {end_time - start_time:.2f} seconds") 
    return grid, nodata_mask


def plot_grid(grid:RasterModelGrid, field_to_plot:str, title:str="Model topography", cmap:str='terrain'):
    """Display the DEM plot without saving."""
    start_time = time.time()
    plt.figure(figsize=(10, 8))

    match field_to_plot:
        case 'status_at_node':
            plt.imshow(grid.status_at_node.reshape(grid.shape), cmap='viridis')
            plt.colorbar(label='Node Status')
        case 'topographic__elevation':
            plt.imshow(grid.at_node['topographic__elevation'].reshape(grid.shape), cmap=cmap)
            plt.colorbar(label='Elevation (m)')
        case _:
            raise ValueError(f"Unknown field to plot: {field_to_plot}")
    plt.colorbar(label='Elevation (m)')
    plt.title(title)
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


    plot_grid(grid, 'status_at_node', title="Node statuses")

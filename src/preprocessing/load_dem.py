import os
import numpy as np
import rasterio
from landlab import RasterModelGrid
import matplotlib.pyplot as plt

# Define base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'figures')

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_dem(dem_path):
    """Load DEM data from a .tif file and return elevation, bounds, and resolution."""
    with rasterio.open(dem_path) as src:
        elevation = src.read(1)  # First band
        bounds = src.bounds
        res = src.res  # (dx, dy)
    return elevation, bounds, res

def create_landlab_grid(elevation, dx, dy):
    """Create a Landlab grid from DEM data."""
    rows, cols = elevation.shape
    grid = RasterModelGrid((rows, cols), xy_spacing=(dx, dy))
    grid.add_field('topographic__elevation', elevation, at='node')
    return grid

def plot_dem(grid, output_path):
    """Plot the DEM and save it."""
    plt.figure(figsize=(10, 8))
    plt.imshow(grid.at_node['topographic__elevation'].reshape(grid.shape), cmap='terrain')
    plt.colorbar(label='Elevation (m)')
    plt.title('Sarez Lake DEM')
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    dem_path = os.path.join(DATA_DIR, 'sarez_dem.tif')  # Adjust filename as needed
    elevation, bounds, (dx, dy) = load_dem(dem_path)
    print(f"DEM loaded. Shape: {elevation.shape}, Resolution: ({dx}, {dy})")

    grid = create_landlab_grid(elevation, dx, dy)
    print(f"Landlab grid created with {grid.number_of_nodes} nodes.")

    plot_path = os.path.join(RESULTS_DIR, 'dem_preview.png')
    plot_dem(grid, plot_path)
    print(f"DEM plot saved to {plot_path}")
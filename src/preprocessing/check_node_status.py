import os
import numpy as np
import rasterio
from rasterio.fill import fillnodata
from landlab import RasterModelGrid

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')


def load_dem(dem_path, mask_path=None):
    """Load DEM data and optionally a watershed mask from .tif files."""
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
    return elevation_filled, bounds, res, nodata, watershed_mask


def create_landlab_grid(elevation, dx, dy, nodata, watershed_mask=None):
    """Create a Landlab grid from DEM data with two outlets (seepage and overflow)."""
    rows, cols = elevation.shape
    grid = RasterModelGrid((rows, cols), xy_spacing=(dx, dy))

    # Convert to float64 and mask no-data values
    elevation_float = elevation.astype(np.float64)
    if nodata is not None:
        elevation_float[elevation == nodata] = 0.0

    # Create no-data mask (if no watershed mask is provided)
    if watershed_mask is None:
        nodata_mask = elevation == nodata
    else:
        # Use the watershed mask to define the boundary (True = outside, False = inside)
        nodata_mask = ~watershed_mask

    # Add elevation field
    grid.add_field('topographic__elevation', elevation_float, at='node')

    # Set no-data nodes to closed
    grid.set_nodata_nodes_to_closed(grid.at_node['topographic__elevation'], nodata)

    # Set valid nodes (inside the watershed) to core nodes
    valid_mask = ~nodata_mask.ravel()
    grid.status_at_node[valid_mask] = grid.BC_NODE_IS_CORE

    # Ensure perimeter nodes of the entire grid are closed
    grid.status_at_node[grid.nodes_at_top_edge] = grid.BC_NODE_IS_CLOSED
    grid.status_at_node[grid.nodes_at_bottom_edge] = grid.BC_NODE_IS_CLOSED
    grid.status_at_node[grid.nodes_at_left_edge] = grid.BC_NODE_IS_CLOSED
    grid.status_at_node[grid.nodes_at_right_edge] = grid.BC_NODE_IS_CLOSED

    # Set two outlet nodes
    # Seepage outlet: At the water level (mean depth 212m), near the dam (row=60, col=149)
    seepage_outlet_node = 60 * cols + 149
    if not nodata_mask.flatten()[seepage_outlet_node]:
        grid.status_at_node[seepage_outlet_node] = grid.BC_NODE_IS_FIXED_VALUE
    else:
        print("Warning: Seepage outlet node is in a no-data area. Please adjust the location.")

    # Overflow outlet: At the dam crest (row=60, col=150), initially closed
    overflow_outlet_node = 60 * cols + 150
    if not nodata_mask.flatten()[overflow_outlet_node]:
        grid.status_at_node[overflow_outlet_node] = grid.BC_NODE_IS_CLOSED
    else:
        print("Warning: Overflow outlet node is in a no-data area. Please adjust the location.")

    return grid, nodata_mask


def check_node_status(grid):
    """Count the number of nodes with each status."""
    closed_nodes = np.sum(grid.status_at_node == 1)  # BC_NODE_IS_CLOSED
    total_nodes = grid.number_of_nodes
    core_nodes = np.sum(grid.status_at_node == 0)  # BC_NODE_IS_CORE
    fixed_value_nodes = np.sum(grid.status_at_node == 4)  # BC_NODE_IS_FIXED_VALUE

    print(f"Total number of nodes: {total_nodes}")
    print(f"Number of nodes with status=1 (BC_NODE_IS_CLOSED): {closed_nodes}")
    print(f"Number of nodes with status=0 (BC_NODE_IS_CORE): {core_nodes}")
    print(f"Number of nodes with status=4 (BC_NODE_IS_FIXED_VALUE): {fixed_value_nodes}")


if __name__ == "__main__":
    dem_path = os.path.join(DATA_DIR, 'sarez1000m.tif')
    mask_path = os.path.join(DATA_DIR, 'sarez_watershed_mask.tif') if os.path.exists(
        os.path.join(DATA_DIR, 'sarez_watershed_mask.tif')) else None

    print("Loading DEM...")
    elevation, bounds, (dx, dy), nodata, watershed_mask = load_dem(dem_path, mask_path)
    print(f"DEM loaded. Shape: {elevation.shape}, Resolution: ({dx}, {dy}), No-data value: {nodata}")

    print("Creating Landlab grid...")
    grid, nodata_mask = create_landlab_grid(elevation, dx, dy, nodata, watershed_mask)
    print(f"Landlab grid created with {grid.number_of_nodes} nodes.")

    print("Checking node status...")
    check_node_status(grid)
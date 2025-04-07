import sys
import os
path_to_package = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','..'))
sys.path.append(path_to_package)

import numpy as np
import matplotlib.pyplot as plt
from landlab.components import FlowAccumulator, ErosionDeposition
from src.preprocessing.load_dem import load_dem, create_landlab_grid

plt.switch_backend('TkAgg')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')


def setup_landlab_model(grid):
    """Set up the Landlab model with flow routing and sediment transport."""
    print("Setting up FlowAccumulator...")
    fa = FlowAccumulator(
        grid,
        depression_finder="DepressionFinderAndRouter",
        flow_director='D8',
        runoff_rate=2
    )
    fa.run_one_step()
    print("Flow accumulation completed.")

    print("Setting up ErosionDeposition...")
    ed = ErosionDeposition(
        grid,
        K=0.000002,
        v_s=0.05,
        F_f=0.8,
        solver='basic'
    )
    print("Sediment transport setup complete.")
    return grid, fa, ed


def run_model(grid, fa, ed, years=10, dt=1.0):
    """Run the model for a specified number of years with timestep dt (years)."""
    print(f"Starting model run for {years} years...")
    n_steps = int(years / dt)
    cols = grid.shape[1]
    seepage_outlet_node = 60 * cols + 149
    overflow_outlet_node = 60 * cols + 150

    # Get base elevations at outlet nodes
    seepage_base_elevation = grid.at_node['topographic__elevation'][seepage_outlet_node]
    overflow_base_elevation = grid.at_node['topographic__elevation'][overflow_outlet_node]

    # Calculate lake level and dam crest elevation
    lake_level = seepage_base_elevation + 212  # Mean depth of Lake Sarez
    dam_crest_elevation = overflow_base_elevation + 650  # Dam height

    print(f"Seepage outlet base elevation (row=60, col=149): {seepage_base_elevation:.2f} m")
    print(f"Overflow outlet base elevation (row=60, col=150): {overflow_base_elevation:.2f} m")
    print(f"Initial lake level: {lake_level:.2f} m")
    print(f"Dam crest elevation: {dam_crest_elevation:.2f} m")

    # Calculate seepage outflow elevation change per timestep
    seepage_rate = 45.1  # m³/s
    seconds_per_year = 31_536_000
    lake_area = 80_000_000  # m² (80 km²)
    volume_per_timestep = seepage_rate * seconds_per_year * dt  # m³
    elevation_change_per_timestep = volume_per_timestep / lake_area  # m
    print(f"Seepage outflow elevation change per timestep: {elevation_change_per_timestep:.2f} m")

    for i in range(n_steps):
        # Simulate seepage outflow by lowering the elevation at the seepage outlet
        grid.at_node['topographic__elevation'][seepage_outlet_node] -= elevation_change_per_timestep

        fa.run_one_step()
        ed.run_one_step(dt)

        # Check lake level (approximate as elevation at seepage outlet + sediment deposition)
        current_lake_level = grid.at_node['topographic__elevation'][seepage_outlet_node] + 212
        if current_lake_level >= dam_crest_elevation and grid.status_at_node[
            overflow_outlet_node] == grid.BC_NODE_IS_CLOSED:
            print(
                f"Step {i + 1}: Lake level ({current_lake_level:.2f} m) has reached dam crest ({dam_crest_elevation:.2f} m). Opening overflow outlet.")
            grid.status_at_node[overflow_outlet_node] = grid.BC_NODE_IS_FIXED_VALUE
            fa = FlowAccumulator(grid, surface='topographic__elevation', flow_director='D8')

        if (i + 1) % (n_steps // 5) == 0:
            print(f"Simulation step {i + 1}/{n_steps}")

    print("Model run complete.")


def plot_results(grid, dx, dy, initial_elevation, nodata_mask):
    """Display the flow accumulation and sediment flux maps within the watershed."""
    print("Preparing plots...")
    cell_area = dx * dy * (111000 ** 2)
    drainage_area_m2 = grid.at_node['drainage_area'] * cell_area
    drainage_area_m2 = drainage_area_m2.reshape(grid.shape)
    drainage_area_m2[nodata_mask] = np.nan

    plt.figure(figsize=(10, 8))
    plt.imshow(drainage_area_m2, cmap='Blues', norm='log')
    plt.colorbar(label='Drainage Area (m²)')
    plt.title(f'Sarez Lake Watershed Flow Accumulation ({int(dx)}m)')
    plt.show()

    elevation_change = grid.at_node['topographic__elevation'] - initial_elevation
    elevation_change = elevation_change.reshape(grid.shape)
    elevation_change[nodata_mask] = np.nan

    plt.figure(figsize=(10, 8))
    plt.imshow(elevation_change, cmap='RdBu', vmin=-0.1, vmax=0.1)
    plt.colorbar(label='Elevation Change (m)')
    plt.title(f'Erosion (Blue) and Deposition (Red) After 10 Years ({int(dx)}m)')
    plt.show()

    print("Plots complete.")


if __name__ == "__main__":
    print("Starting process...")
    dem_path = os.path.join(DATA_DIR, 'sarez1000m.tif')
    mask_path = os.path.join(DATA_DIR, 'sarez_watershed_mask.tif') if os.path.exists(
        os.path.join(DATA_DIR, 'sarez_watershed_mask.tif')) else None

    print("Loading DEM...")
    elevation, bounds, (dx, dy), nodata, watershed_mask = load_dem(dem_path, mask_path)
    print("Creating Landlab grid...")
    grid, nodata_mask = create_landlab_grid(elevation, dx, dy, nodata, watershed_mask)
    print(f"Grid loaded with {grid.number_of_nodes} nodes.")

    initial_elevation = grid.at_node['topographic__elevation'].copy()

    grid, fa, ed = setup_landlab_model(grid)

    run_model(grid, fa, ed, years=10, dt=1.0)

    plot_results(grid, dx, dy, initial_elevation, nodata_mask)
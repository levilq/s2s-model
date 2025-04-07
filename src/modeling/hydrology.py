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
        K=0.0001,  # Increased erodibility for mountainous terrain
        v_s=0.05,  # Settling velocity for fine sediment (silt)
        F_f=0.5,   # 50% fine sediment, reflecting a mix of coarse and fine
        m_sp=1.0,  # Discharge exponent
        n_sp=1.0,  # Slope exponent
        solver='basic'
    )
    print("Sediment transport setup complete.")
    return grid, fa, ed

def run_model(grid, fa, ed, seepage_outlet_node, overflow_outlet_node, years=10, dt=0.1):
    """Run the model for a specified number of years with timestep dt (years)."""
    print(f"Starting model run for {years} years with timestep {dt} years...")
    n_steps = int(years / dt)

    # Get base elevations at outlet nodes
    seepage_base_elevation = grid.at_node['topographic__elevation'][seepage_outlet_node]
    overflow_base_elevation = grid.at_node['topographic__elevation'][overflow_outlet_node]

    # Calculate lake level and dam crest elevation
    lake_level = seepage_base_elevation + 212  # Mean depth of Lake Sarez
    dam_crest_elevation = overflow_base_elevation + 650  # Dam height

    print(f"Seepage outlet base elevation: {seepage_base_elevation:.2f} m")
    print(f"Overflow outlet base elevation: {overflow_base_elevation:.2f} m")
    print(f"Initial lake level: {lake_level:.2f} m")
    print(f"Dam crest elevation: {dam_crest_elevation:.2f} m")

    # Calculate seepage outflow elevation change per timestep
    seepage_rate = 45.1  # m³/s
    seconds_per_year = 31_536_000
    lake_area = 80_000_000  # m² (80 km²)
    volume_per_timestep = seepage_rate * seconds_per_year * dt  # m³
    elevation_change_per_timestep = volume_per_timestep / lake_area  # m
    print(f"Seepage outflow elevation change per timestep: {elevation_change_per_timestep:.2f} m")

    # Track total erosion and deposition
    total_erosion = 0.0
    total_deposition = 0.0

    for i in range(n_steps):
        # Store elevation at the start of the timestep
        elevation_before = grid.at_node['topographic__elevation'].copy()

        # Simulate seepage outflow by lowering the elevation at the seepage outlet
        grid.at_node['topographic__elevation'][seepage_outlet_node] -= elevation_change_per_timestep

        # Run flow accumulation and erosion/deposition
        fa.run_one_step()
        ed.run_one_step(dt)

        # Calculate elevation change due to erosion and deposition
        elevation_after = grid.at_node['topographic__elevation'].copy()
        elevation_change = elevation_after - elevation_before

        # Filter for core nodes first
        core_nodes = grid.status_at_node == grid.BC_NODE_IS_CORE
        elevation_change_core = elevation_change[core_nodes]

        # Sum erosion (negative change) and deposition (positive change) for core nodes
        total_erosion += np.sum(elevation_change_core[elevation_change_core < 0])
        total_deposition += np.sum(elevation_change_core[elevation_change_core > 0])

        # Check lake level (approximate as elevation at seepage outlet + sediment deposition)
        current_lake_level = grid.at_node['topographic__elevation'][seepage_outlet_node] + 212
        if current_lake_level >= dam_crest_elevation and grid.status_at_node[overflow_outlet_node] == grid.BC_NODE_IS_CLOSED:
            print(f"Step {i + 1}: Lake level ({current_lake_level:.2f} m) has reached dam crest ({dam_crest_elevation:.2f} m). Opening overflow outlet.")
            grid.status_at_node[overflow_outlet_node] = grid.BC_NODE_IS_FIXED_VALUE
            fa = FlowAccumulator(
                grid,
                depression_finder="DepressionFinderAndRouter",
                flow_director='D8',
                runoff_rate=2
            )

        if (i + 1) % (n_steps // 5) == 0:
            print(f"Simulation step {i + 1}/{n_steps}")
            print(f"Cumulative erosion: {total_erosion:.2f} m")
            print(f"Cumulative deposition: {total_deposition:.2f} m")

    print("Model run complete.")
    print(f"Total cumulative erosion: {total_erosion:.2f} m")
    print(f"Total cumulative deposition: {total_deposition:.2f} m")

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
    plt.imshow(elevation_change, cmap='RdBu_r', vmin=-1.0, vmax=1.0)  # Reversed colormap: erosion (negative) in red, deposition (positive) in blue
    plt.colorbar(label='Elevation Change (m)')
    plt.title(f'Erosion (Red) and Deposition (Blue) After {years} Years ({int(dx)}m)')
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
    grid, nodata_mask, seepage_outlet_node, overflow_outlet_node = create_landlab_grid(elevation, dx, dy, nodata, watershed_mask)
    print(f"Grid loaded with {grid.number_of_nodes} nodes.")

    initial_elevation = grid.at_node['topographic__elevation'].copy()

    grid, fa, ed = setup_landlab_model(grid)

    years = 100  # Changed to 100 years as per your modification
    run_model(grid, fa, ed, seepage_outlet_node, overflow_outlet_node, years=years, dt=0.1)

    plot_results(grid, dx, dy, initial_elevation, nodata_mask)
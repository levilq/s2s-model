# src/modeling/hydrology.py
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
        F_f=0,   # 50% fine sediment, reflecting a mix of coarse and fine
        m_sp=0.5,  # Discharge exponent
        n_sp=1.0,  # Slope exponent
        solver='basic'
    )
    print("Sediment transport setup complete.")
    return grid, fa, ed

def run_model(grid, fa, ed, seepage_outlet_node, overflow_outlet_node, lake_mask, years=100, dt=0.1):
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

    # Calculate seepage outflow (for informational purposes, but not used to lower elevation)
    seepage_rate = 45.1  # m³/s
    seconds_per_year = 31_536_000
    lake_area = 80_000_000  # m² (80 km²)
    volume_per_timestep = seepage_rate * seconds_per_year * dt  # m³
    elevation_change_per_timestep = volume_per_timestep / lake_area  # m
    print(f"Seepage outflow elevation change per timestep (for reference): {elevation_change_per_timestep:.2f} m")

    # Track total erosion and deposition
    total_erosion = 0.0
    total_deposition = 0.0

    # Store initial elevation to calculate total elevation change
    initial_elevation = grid.at_node['topographic__elevation'].copy()

    # Convert lake mask to node array for easier indexing
    lake_mask_nodes = lake_mask.flatten() if lake_mask is not None else np.zeros_like(initial_elevation, dtype=bool)

    # Track sediment flux over time
    sediment_density = 2650  # kg/m³
    sediment_flux_history = []  # Sediment flux (m³/s) at each timestep
    cumulative_sediment_volume = []  # Cumulative sediment volume (m³)
    cumulative_volume = 0.0

    for i in range(n_steps):
        # Store elevation at the start of the timestep
        elevation_before = grid.at_node['topographic__elevation'].copy()

        # Run flow accumulation and erosion/deposition
        fa.run_one_step()
        ed.run_one_step(dt)

        # Calculate elevation change due to erosion and deposition
        elevation_after = grid.at_node['topographic__elevation'].copy()
        elevation_change = elevation_after - elevation_before

        #This does not make sense: 
        """ # Update the elevation field
        grid.at_node['topographic__elevation'][:] = elevation_before + elevation_change"""

        # Prevent elevation in the lake from decreasing below the initial elevation
        lake_nodes = (lake_mask_nodes == 1)
        current_elevation = grid.at_node['topographic__elevation']
        below_initial = current_elevation < initial_elevation
        nodes_to_correct = lake_nodes & below_initial
        grid.at_node['topographic__elevation'][nodes_to_correct] = initial_elevation[nodes_to_correct]

        # Ensure the seepage outlet node allows flow but doesn't induce erosion
        grid.at_node['topographic__elevation'][seepage_outlet_node] = min(
            grid.at_node['topographic__elevation'][seepage_outlet_node], lake_level
        )

        # Filter for core nodes outside the lake for erosion/deposition tracking
        core_nodes_outside_lake = (grid.status_at_node == grid.BC_NODE_IS_CORE) & (lake_mask_nodes == 0)
        elevation_change_core = elevation_change[core_nodes_outside_lake]

        # Sum erosion (negative change) and deposition (positive change) for core nodes outside the lake
        total_erosion += np.sum(elevation_change_core[elevation_change_core < 0])
        total_deposition += np.sum(elevation_change_core[elevation_change_core > 0])

        # Calculate sediment flux at the seepage outlet for this timestep
        outflux_mass = grid.at_node['sediment__outflux'][seepage_outlet_node]  # kg/s
        outflux_volume_rate = outflux_mass / sediment_density  # m³/s
        sediment_flux_history.append(outflux_volume_rate)

        # Accumulate the sediment volume for this timestep (m³)
        outflux_volume = outflux_volume_rate * (dt * seconds_per_year)  # m³
        cumulative_volume += outflux_volume
        cumulative_sediment_volume.append(cumulative_volume)

        # Check lake level for overflow
        current_lake_level = grid.at_node['topographic__elevation'][seepage_outlet_node]
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
            print(f"Cumulative erosion (outside lake): {total_erosion:.2f} m")
            print(f"Cumulative deposition (outside lake): {total_deposition:.2f} m")

    print("Model run complete.")
    print(f"Total cumulative erosion (outside lake): {total_erosion:.2f} m")
    print(f"Total cumulative deposition (outside lake): {total_deposition:.2f} m")

    # Calculate sediment influx (erosion outside the lake)
    elevation_change_total = grid.at_node['topographic__elevation'] - initial_elevation
    lake_nodes = (lake_mask_nodes == 1)
    elevation_change_total[lake_nodes & (elevation_change_total < 0)] = 0.0
    elevation_change_total_2d = elevation_change_total.reshape(grid.shape)

    if lake_mask is not None:
        lake_mask_nodes = lake_mask.flatten()
        outside_lake_nodes = (lake_mask_nodes == 0) & (grid.status_at_node == grid.BC_NODE_IS_CORE)
    else:
        outside_lake_nodes = (grid.status_at_node == grid.BC_NODE_IS_CORE)

    erosion = elevation_change_total[outside_lake_nodes]
    erosion = erosion[erosion < 0]
    total_erosion_depth = np.sum(np.abs(erosion))
    cell_area = (111000 * 0.0045) ** 2
    sediment_influx_volume = total_erosion_depth * cell_area
    print(f"Sediment Influx (from erosion outside the lake): {sediment_influx_volume:.2f} m³")

    # Calculate sediment outflux at the seepage outlet
    outflux_mass = grid.at_node['sediment__outflux'][seepage_outlet_node]
    total_outflux_mass = outflux_mass * (years * 31_536_000)
    sediment_outflux_volume = total_outflux_mass / sediment_density
    print(f"Sediment Outflux (at seepage outlet): {sediment_outflux_volume:.2f} m³")

    # Calculate net sediment flux in the lake
    net_sediment_volume = sediment_influx_volume - sediment_outflux_volume
    print(f"Net Sediment Flux in Lake Sarez (influx - outflux): {net_sediment_volume:.2f} m³")

    # Calculate total deposition within the lake
    lake_nodes = (lake_mask_nodes == 1)
    deposition_in_lake = elevation_change_total[lake_nodes]
    deposition_in_lake = deposition_in_lake[deposition_in_lake > 0]
    total_deposition_depth_in_lake = np.sum(deposition_in_lake)
    total_deposition_volume_in_lake = total_deposition_depth_in_lake * cell_area
    print(f"Total Deposition Volume in Lake Sarez: {total_deposition_volume_in_lake:.2f} m³")

    return elevation_change_total, grid.at_node['sediment__outflux'], net_sediment_volume, sediment_flux_history, cumulative_sediment_volume

def plot_results(grid, dx, dy, initial_elevation, nodata_mask, lake_mask, elevation_change_total, sediment_outflux, seepage_outlet_node, years, net_sediment_volume, sediment_flux_history, cumulative_sediment_volume, dt):
    """Display the flow accumulation, erosion/deposition, sediment influx/outflux, and net sediment flux maps within the watershed."""
    print("Preparing plots...")
    cell_area = dx * dy * (111000 ** 2)
    drainage_area_m2 = grid.at_node['drainage_area'] * cell_area
    drainage_area_m2 = drainage_area_m2.reshape(grid.shape)
    drainage_area_m2[nodata_mask] = np.nan

    # Plot flow accumulation
    plt.figure(figsize=(10, 8))
    plt.imshow(drainage_area_m2, cmap='Blues', norm='log')
    plt.colorbar(label='Drainage Area (m²)')
    plt.title(f'Sarez Lake Watershed Flow Accumulation ({int(dx * 111000)}m)')
    plt.xlabel('X (grid cells)')
    plt.ylabel('Y (grid cells)')
    plt.show()

    # Plot elevation change (erosion/deposition)
    elevation_change = grid.at_node['topographic__elevation'] - initial_elevation
    elevation_change = elevation_change.reshape(grid.shape)
    elevation_change[nodata_mask] = np.nan

    plt.figure(figsize=(10, 8))
    plt.imshow(elevation_change, cmap='RdBu_r')
    plt.colorbar(label='Elevation Change (m)')
    plt.title(f'Elevation change after evolving the landscape for {years} Years ({int(dx * 111000)}m)')
    plt.xlabel('X (grid cells)')
    plt.ylabel('Y (grid cells)')
    plt.show()

    # Plot sediment influx map (elevation change within the lake, zoomed to lake extent)
    elevation_change_total_2d = elevation_change_total.reshape(grid.shape)
    elevation_change_total_2d[nodata_mask] = np.nan

    # Debug: Check elevation changes within the lake
    if lake_mask is not None:
        lake_mask_nodes = lake_mask.flatten()
        lake_nodes = (lake_mask_nodes == 1)
        deposition_in_lake = elevation_change_total[lake_nodes]
        print(f"Debug: Elevation changes in lake (min, max, mean): {deposition_in_lake.min():.4f}, {deposition_in_lake.max():.4f}, {deposition_in_lake.mean():.4f}")
        print(f"Debug: Number of positive elevation changes in lake: {np.sum(deposition_in_lake > 0)}")

    if lake_mask is not None:
        # Create a mask to isolate the lake area (lake = 1, outside = NaN)
        lake_mask_2d = lake_mask.astype(np.float32)
        lake_mask_2d[lake_mask_2d == 0] = np.nan  # Outside lake becomes NaN
        lake_mask_2d[lake_mask_2d == 1] = 1.0    # Inside lake remains 1
        influx_map = elevation_change_total_2d * lake_mask_2d

        # Debug: Check the influx map values within the lake
        lake_indices = np.where(lake_mask == 1)
        if len(lake_indices[0]) > 0:
            influx_values = influx_map[lake_indices]
            print(f"Debug: Influx map values in lake (min, max, mean): {np.nanmin(influx_values):.4f}, {np.nanmax(influx_values):.4f}, {np.nanmean(influx_values):.4f}")

        # Set plot extent to zoom into the lake
        lake_indices = np.where(lake_mask == 1)
        if len(lake_indices[0]) > 0:
            min_row, max_row = lake_indices[0].min(), lake_indices[0].max()
            min_col, max_col = lake_indices[1].min(), lake_indices[1].max()
            buffer = 1
            min_row = max(0, min_row - buffer)
            max_row = min(grid.shape[0] - 1, max_row + buffer)
            min_col = max(0, min_col - buffer)
            max_col = min(grid.shape[1] - 1, max_col + buffer)
            cell_size_m = int(dx * 111000)
            min_x_m = min_col * cell_size_m
            max_x_m = max_col * cell_size_m
            min_y_m = min_row * cell_size_m
            max_y_m = max_row * cell_size_m
        else:
            print("Warning: Lake mask contains no lake area (all zeros). Showing full extent.")
            min_row, max_row = 0, grid.shape[0] - 1
            min_col, max_col = 0, grid.shape[1] - 1
            min_x_m, max_x_m = 0, grid.shape[1] * cell_size_m
            min_y_m, max_y_m = 0, grid.shape[0] * cell_size_m
    else:
        influx_map = elevation_change_total_2d
        min_row, max_row = 0, grid.shape[0] - 1
        min_col, max_col = 0, grid.shape[1] - 1
        min_x_m, max_x_m = 0, grid.shape[1] * cell_size_m
        min_y_m, max_y_m = 0, grid.shape[0] * cell_size_m

    # Adjust vmin and vmax based on actual deposition values
    if lake_mask is not None:
        deposition_values = influx_map[lake_indices]
        if np.sum(deposition_values > 0) > 0:
            vmin = 0.0
            vmax = np.nanmax(deposition_values) * 1.1  # Slightly above max for visibility
            print(f"Debug: Adjusted vmin, vmax for influx plot: {vmin:.4f}, {vmax:.4f}")
        else:
            vmin, vmax = 0.0, 1.0
            print("Debug: No positive deposition values in lake, using default vmin, vmax: 0.0, 1.0")
    else:
        vmin, vmax = 0.0, 1.0

    plt.figure(figsize=(10, 8))
    plt.imshow(influx_map, cmap='Blues', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Deposition (m)')
    plt.title(f'Sediment Influx Effects in Lake Sarez After {years} Years ({int(dx * 111000)}m)')
    plt.xlim(min_col, max_col)
    plt.ylim(max_row, min_row)
    plt.xticks(ticks=np.linspace(min_col, max_col, num=5), labels=np.linspace(min_x_m, max_x_m, num=5).astype(int))
    plt.yticks(ticks=np.linspace(min_row, max_row, num=5), labels=np.linspace(min_y_m, max_y_m, num=5).astype(int))
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.show()

    # Plot sediment outflux map
    sediment_outflux_2d = sediment_outflux.reshape(grid.shape)
    sediment_outflux_2d[nodata_mask] = np.nan
    rows, cols = grid.shape
    seepage_row, seepage_col = divmod(seepage_outlet_node, cols)
    plt.figure(figsize=(10, 8))
    plt.imshow(sediment_outflux_2d, cmap='Blues', norm='log')
    plt.colorbar(label='Sediment Outflux (kg/s)')
    plt.scatter(seepage_col, seepage_row, color='red', s=100, label='Seepage Outlet', marker='x')
    plt.legend()
    plt.title(f'Sediment Outflux After {years} Years ({int(dx * 111000)}m)')
    plt.xlabel('X (grid cells)')
    plt.ylabel('Y (grid cells)')
    plt.show()

    # Plot net sediment flux within the lake
    net_flux_map = elevation_change.reshape(grid.shape)
    net_flux_map[nodata_mask] = np.nan
    if lake_mask is not None:
        outside_lake_mask = (lake_mask == 0).astype(np.float32)
        outside_lake_mask[outside_lake_mask == 1] = np.nan
        net_flux_map = net_flux_map * outside_lake_mask
    net_flux_map = np.where(net_flux_map > 0, net_flux_map, np.nan)
    plt.figure(figsize=(10, 8))
    plt.imshow(net_flux_map, cmap='Greens', vmin=0.0, vmax=1.0)
    plt.colorbar(label='Net Sediment Deposition (m)')
    plt.title(f'Net Sediment Flux (Deposition) in Lake Sarez After {years} Years ({int(dx * 111000)}m)')
    plt.xlabel('X (grid cells)')
    plt.ylabel('Y (grid cells)')
    plt.show()

    # Plot cumulative sediment flux over time
    time_steps = np.arange(0, years, dt)
    plt.figure(figsize=(10, 8))
    plt.plot(time_steps, cumulative_sediment_volume, label='Cumulative Sediment Outflux', color='blue')
    plt.xlabel('Time (years)')
    plt.ylabel('Cumulative Sediment Volume (m³)')
    plt.title('Cumulative Sediment Outflux at Seepage Outlet Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    print("Plots complete.")

if __name__ == "__main__":
    print("Starting process...")
    dem_path = os.path.join(DATA_DIR, 'sarez500m.tif')
    mask_path = os.path.join(DATA_DIR, 'sarez_watershed_mask.tif') if os.path.exists(
        os.path.join(DATA_DIR, 'sarez_watershed_mask.tif')) else None
    lake_mask_path = os.path.join(DATA_DIR, 'sarez_lake_mask_500m.tif')

    print("Loading DEM...")
    elevation, bounds, (dx, dy), nodata, watershed_mask, lake_mask = load_dem(dem_path, mask_path, lake_mask_path)
    print("Creating Landlab grid...")
    grid, nodata_mask, seepage_outlet_node, overflow_outlet_node = create_landlab_grid(elevation, dx, dy, nodata, watershed_mask)
    print(f"Grid loaded with {grid.number_of_nodes} nodes.")

    initial_elevation = grid.at_node['topographic__elevation'].copy()

    grid, fa, ed = setup_landlab_model(grid)

    years = 100
    dt = 0.1  # Define dt here to pass to plot_results
    elevation_change_total, sediment_outflux, net_sediment_volume, sediment_flux_history, cumulative_sediment_volume = run_model(
        grid, fa, ed, seepage_outlet_node, overflow_outlet_node, lake_mask, years=years, dt=dt
    )

    plot_results(
        grid, dx, dy, initial_elevation, nodata_mask, lake_mask, elevation_change_total,
        sediment_outflux, seepage_outlet_node, years, net_sediment_volume, sediment_flux_history, cumulative_sediment_volume, dt
    )

    print(f"Final Net Sediment Volume in Lake Sarez: {net_sediment_volume:.2f} m³")
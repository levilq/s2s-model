import os
import warnings

from landlab import RasterModelGrid
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.fill import fillnodata

try:
    import utm
except ImportError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "utm"])
    import utm

from landlab.components import FlowAccumulator, ErosionDeposition, SinkFiller

from utils import ProcessLogger, resampleDEM

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'out')
GRID_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'grid')
DF_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'df')
STATS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'stats')
TRENDS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'trends')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(GRID_OUTPUT_DIR):
    os.makedirs(GRID_OUTPUT_DIR)
if not os.path.exists(DF_OUTPUT_DIR):
    os.makedirs(DF_OUTPUT_DIR)
if not os.path.exists(STATS_OUTPUT_DIR):
    os.makedirs(STATS_OUTPUT_DIR)
if not os.path.exists(TRENDS_OUTPUT_DIR):
    os.makedirs(TRENDS_OUTPUT_DIR)

class SourceToSinkSimulator:
    def __init__(self, path_to_topography: str, path_to_precipitation: str = None, path_to_watershed_mask: str = None, path_to_sink_mask: str = None, path_to_depth_map: str = None, runtime_plotting: bool = False):
        self.path_to_topography = path_to_topography
        self.path_to_precipitation = path_to_precipitation
        self.path_to_watershed_mask = path_to_watershed_mask
        self.path_to_sink_mask = path_to_sink_mask
        self.path_to_depth_map = path_to_depth_map
        self.grid = None
        self.nodata_mask = None
        self.valid_data_mask = None
        self.no_data_value = None
        self.shape = None
        self.model_x_spacing = None
        self.model_y_spacing = None
        self.flow_accumulator = None
        self.depression_finder = None
        self.erosion_deposition_model = None
        self.model_runtime = None
        self.initial_topography = None
        self.initial_mean_elevation = None
        self.runtime_plotting = runtime_plotting
        self.save_step_results = True
        self.outlet_id = None
        self.depth_map = None
        self.logger = ProcessLogger(os.path.join(OUTPUT_DIR, 'process.log'))
        self.logger.log("SourceToSinkSimulator initialized.")
        self.trends = {
            'flow__link_to_receiver_node': [],
            'sediment__flux': [],
            'sediment__flux_lake': [],
            'sediment__influx': [],
            'sediment__outflux': [],
            'sediment__influx_lake': [],
            'sediment__outflux_lake': [],
            'surface_water__discharge': [],
            'surface_water__discharge_non_lake': [],
            'surface_water__discharge_outlet': [],
            'topographic__elevation': [],
            'topographic__steepest_slope': [],
            'elevation_difference_lake': [],
            'topography_change': []
        }
        self.trend_years = []

    def loadRaster(self, raster_path: str, is_mask: bool = False, is_dem: bool = False):
        """Load a raster file."""
        try:
            with rasterio.open(raster_path) as src:
                if src.crs.is_geographic:
                    dst_path = raster_path.replace('.tif', '_utm.tif')
                    self.reprojectToUtm(src, dst_path)
                    src = rasterio.open(dst_path)
                    print(f"Reprojected raster to UTM and saved as {dst_path}.")
                if self.grid and src.shape != self.grid.shape:
                    print(f"grid shape: {self.grid.shape}, raster shape: {src.shape}")
                    print(f"Warning: The resolution of the {raster_path} raster does not match the DEM resolution.")
                    print("Resampling the raster to match the DEM resolution.")
                    path_to_raster = resampleDEM(raster_path,
                                                 up_scale_factor=src.res[0] / self.model_resolution)
                    src = rasterio.open(path_to_raster)

                if is_mask:
                    data = src.read(1).astype(np.int32)
                else:
                    data = src.read(1).astype(np.float64)

                if is_dem:
                    self.model_x_spacing = src.res[0]
                    self.model_y_spacing = src.res[1]
        except Exception as e:
            print(f"Error loading raster: {e}")
            return None

        return data

    def createRasterModelGrid(self):
        """Create a RasterModelGrid from a DEM file."""
        elevation_data = self.loadRaster(self.path_to_topography, is_dem=True)
        if elevation_data is None:
            raise ValueError("Elevation data not found or invalid. Please check the path.")
        self.shape = elevation_data.shape

        if self.path_to_watershed_mask:
            watershed_mask = self.loadRaster(self.path_to_watershed_mask, is_mask=True)
            if watershed_mask is not None:
                elevation_data[watershed_mask == 0] = self.no_data_value
                self.valid_data_mask = watershed_mask
                self.nodata_mask = elevation_data == self.no_data_value
                self.logger.log(f"Watershed mask loaded from {self.path_to_watershed_mask}.")
            else:
                raise ValueError("Watershed mask not found or invalid. Please check the path.")
        else:
            raise ValueError("Watershed mask not provided. Please provide a valid watershed mask.")

        self.grid = RasterModelGrid(elevation_data.shape, xy_spacing=(self.model_x_spacing, self.model_y_spacing))
        self.grid.add_field('topographic__elevation', elevation_data, at='node', units='m')
        self.grid.add_field('watershed_mask', self.valid_data_mask, at='node', units='binary')

        self.logger.log("RasterModelGrid created successfully.")
        self.logger.log(f"Model resolution (x,y): {self.model_x_spacing, self.model_y_spacing} m")
        self.logger.log(f"Model grid shape: {self.grid.shape}")
        self.logger.log(f"Model grid size: {self.grid.number_of_nodes} nodes")
        self.logger.log(f"Model grid cell size: {self.grid.dx} m x {self.grid.dy} m")
        self.logger.log(f"Model grid cell area: {self.grid.dx * self.grid.dy} m2")
        self.logger.log(f"Model grid extent: {self.grid.extent}")
        self.logFieldStats('topographic__elevation')

        if self.path_to_precipitation:
            precipitation_data = self.loadRaster(self.path_to_precipitation)
            if precipitation_data is not None:
                precipitation = precipitation_data / 1000.0
                precipitation[self.nodata_mask] = self.no_data_value
                self.grid.add_field('water__unit_flux_in', precipitation, at='node', clobber=True, units='m/year')
                self.logger.log(f"Precipitation rates set from {self.path_to_precipitation}.")
                self.logFieldStats('water__unit_flux_in')
            else:
                warnings.warn("Precipitation data not found or invalid. Proceeding without it. to accumulate flow rainfall rate of 2m/year is used.")

        self.initial_topography = self.grid.at_node['topographic__elevation'].copy()

        if self.path_to_sink_mask:
            lake_mask = self.loadRaster(self.path_to_sink_mask, is_mask=True)
            if lake_mask is not None:
                self.grid.add_field('sink_mask', lake_mask, at='node', units='binary')
                self.logger.log(f"Sink mask loaded from {self.path_to_sink_mask}.")
                self.plotDataArray(self.grid.at_node['sink_mask'], title="Sink Mask", save_path=os.path.join(OUTPUT_DIR, 'sink_mask.png'))
        else:
            warnings.warn("No sink mask provided. Proceeding without it. Some plots may not be available.")

        if self.path_to_depth_map:
            depth_data = self.loadRaster(self.path_to_depth_map)
            if depth_data is not None:
                self.depth_map = depth_data
                self.depth_map[self.nodata_mask] = np.nan
                self.logger.log(f"Depth map loaded from {self.path_to_depth_map}.")
            else:
                warnings.warn("Depth map not found or invalid. Proceeding without it. Lake fill time calculation using depth map will not be available.")

        if self.runtime_plotting:
            self.plotFieldData('topographic__elevation')
            self.plotDataArray(self.nodata_mask, title="NoData Mask")

    def reprojectToUtm(self, src, dst_path):
        """Reprojects a raster to UTM project coordinate system."""
        if src.crs.is_geographic:
            lon = (src.bounds.left + src.bounds.right) / 2
            lat = (src.bounds.top + src.bounds.bottom) / 2
            utm_zone = utm.from_latlon(lat, lon)
            utm_crs = f"EPSG:326{utm_zone[2]}" if src.crs.to_epsg() == 4326 else f"EPSG:327{utm_zone[2]}"
        else:
            print("Raster is already in UTM coordinates.")
            return

        transform, width, height = calculate_default_transform(
            src.crs, utm_crs, src.width, src.height, *src.bounds
        )
        metadata = src.meta.copy()
        metadata.update({
            'crs': utm_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dst_path, 'w', **metadata) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=utm_crs,
                    resampling=Resampling.nearest
                )

    def setWatershedBoundaryConditions(self):
        """Set watershed boundary conditions for the DEM."""
        def log(outlet_id):
            rows, cols = self.grid.shape
            outlet_row, outlet_col = self.grid.node_x[self.outlet_id], self.grid.node_y[self.outlet_id]
            self.logger.log(f"Outlet node ID: {self.outlet_id}")
            self.logger.log(f"Outlet grid position: (row={outlet_row}, col={outlet_col})")
            print(f"Outlet node ID: {self.outlet_id}")
            print(f"Outlet grid position: (row={outlet_row}, col={outlet_col})")

            elevation = self.grid.at_node['topographic__elevation']
            core_elevations = elevation[self.grid.core_nodes]
            min_elev = np.min(core_elevations)
            min_node = self.grid.core_nodes[np.argmin(core_elevations)]
            max_elev = np.max(core_elevations)
            max_node = self.grid.core_nodes[np.argmax(core_elevations)]

            print(f"Minimum elevation in watershed: {min_elev} m at node {min_node}")
            print(f"Maximum elevation in watershed: {max_elev} m at node {max_node}")
            print(
                f"Outlet node: {self.outlet_id}, elevation: {self.grid.at_node['topographic__elevation'][self.outlet_id]} m")

            self.logger.log("Watershed boundary conditions set successfully.")
            self.logger.log(f"Outlet node ID: {outlet_id}")
            self.logger.log(f"Outlet grid position: (row={outlet_row}, col={outlet_col})")
            self.logger.log(
                f"Outlet spatial coordinates: ({self.grid.node_x[outlet_id]}, {self.grid.node_y[outlet_id]})")
            self.logger.log(f"Outlet node elevation: {self.grid.at_node['topographic__elevation'][outlet_id]}")
            self.logger.log(f"Outlet node status: {self.grid.status_at_node[outlet_id]}")
            self.logger.log(f"Number of closed nodes: {np.sum(self.grid.status_at_node == 4)}")
            self.logger.log(f"Number of core nodes: {np.sum(self.grid.status_at_node == 0)}")
            self.logger.log(f"Number of nodes with fixed value: {np.sum(self.grid.status_at_node == 1)}")
            self.logger.log(f"Number of nodes with fixed gradient: {np.sum(self.grid.status_at_node == 2)}")
            self.logger.log(f"Number of looped nodes: {np.sum(self.grid.status_at_node == 3)}")
            self.logger.log(f"Topography stats after setting watershed boundary conditions:")
            self.logFieldStats('topographic__elevation')

        if self.grid is None:
            raise ValueError("Grid not created. Call createRasterModelGrid() first.")

        self.logger.log("Setting watershed boundary conditions...")

        self.grid.set_nodata_nodes_to_closed(self.grid.at_node['topographic__elevation'], self.no_data_value)
        try:
            outlet_id = self.grid.set_watershed_boundary_condition(
                self.grid.field_values('topographic__elevation'),
                nodata_value=self.no_data_value,
                remove_disconnected=True,
                return_outlet_id=True
            )
            self.outlet_id = outlet_id
            log(outlet_id=outlet_id)
        except Exception as e:
            print(f"Error setting watershed boundary conditions: {e}")
            print("Seems like there are multiple cells with the lowest elevation.")
            print("Trying to select one of the them as the outlet.")
            min_elev = np.min(self.grid.at_node['topographic__elevation'][not np.isnan(self.grid.at_node['topographic__elevation'])])
            node_id = np.where(self.grid.at_node['topographic__elevation'] == min_elev)[0]
            print(f"Setting outlet to node {node_id} with elevation {min_elev} m")
            outlet_id = self.grid.set_watershed_boundary_condition_outlet_id(
                node_id, self.grid.field_values("topographic__elevation")
            )
            self.outlet_id = outlet_id
            log(outlet_id=outlet_id)

        if not outlet_id is None:
            self.logger.log("Watershed boundary conditions set successfully.")

        self.plotOutletOnTopography(save_path=os.path.join(OUTPUT_DIR, 'outlet_location.png'))
        self.plotDataArray(self.valid_data_mask, title="Watershed Area (Valid Data Mask)",
                           save_path=os.path.join(OUTPUT_DIR, 'watershed_area.png'))

    def plotOutletOnTopography(self, save_path: str = None):
        """Plot the outlet on the topographic elevation map with lake mask overlay."""
        if self.grid is None:
            raise ValueError("Grid not created. Call createRasterModelGrid() first.")
        if self.outlet_id is None:
            raise ValueError("Outlet not set. Call setWatershedBoundaryConditions() first.")

        outlet_x, outlet_y = self.grid.node_x[self.outlet_id], self.grid.node_y[self.outlet_id]
        print(f"Outlet coordinates: ({outlet_x}, {outlet_y})")
        rows, cols = self.grid.shape
        outlet_row, outlet_col = divmod(self.outlet_id, cols)
        self.logger.log(
            f"Plotting outlet at: node_id={self.outlet_id}, grid position=(row={outlet_row}, col={outlet_col})")

        plt.figure(figsize=(10, 10))
        topo = self.grid.at_node['topographic__elevation'].reshape(self.grid.shape)
        topo[self.nodata_mask] = np.nan
        plt.imshow(topo, cmap='terrain')
        plt.colorbar(label='Elevation (m)')

        if self.grid.has_field('sink_mask'):
            lake_contour = plt.contour(self.grid.at_node["sink_mask"].reshape(self.grid.shape), levels=[0.5], colors='blue', linestyles='--', linewidths=2)

        outlet_point = plt.scatter(outlet_col, outlet_row, color='red', s=100, marker='x')
        plt.title('Outlet Location on Topography')
        plt.xlabel('X (grid cells)')
        plt.ylabel('Y (grid cells)')

        if save_path:
            plt.savefig(save_path)
            self.logger.log(f"Outlet plot saved to {save_path}")
        plt.close()

    def plotDataArray(self, data_array: np.ndarray, title: str = None, save_path: str = None, cmap: str = 'viridis',
                      label: str = 'Value', crop_to_lake: bool = False):
        """Plot a data array, optionally saving to a file, with option to crop to lake extent."""
        if self.grid is None:
            raise ValueError("Grid not created. Call createRasterModelGrid() first.")
        plt.figure(figsize=(10, 10))
        data_2d = data_array.reshape(self.grid.shape) if data_array.ndim == 1 else data_array

        if not "Watershed area" in str(title):
            if np.issubdtype(data_2d.dtype, np.integer):
                data_2d = data_2d.astype(np.float64)
            data_2d[self.nodata_mask] = np.nan

        if crop_to_lake and self.grid.has_field('sink_mask'):
            lake_mask_2d = self.grid.at_node['sink_mask'].reshape(self.grid.shape)
            lake_indices = np.where(lake_mask_2d == 1)
            if len(lake_indices[0]) == 0:
                self.logger.log("No lake area found in sink_mask. Plotting full extent.")
            else:
                min_row, max_row = np.min(lake_indices[0]), np.max(lake_indices[0])
                min_col, max_col = np.min(lake_indices[1]), np.max(lake_indices[1])
                buffer = 5
                min_row = max(0, min_row - buffer)
                max_row = min(self.grid.shape[0] - 1, max_row + buffer)
                min_col = max(0, min_col - buffer)
                max_col = min(self.grid.shape[1] - 1, max_col + buffer)
                data_2d = data_2d[min_row:max_row + 1, min_col:max_col + 1]
                self.logger.log(f"Cropped plot to lake extent: rows {min_row} to {max_row}, cols {min_col} to {max_col}")
                plt.imshow(data_2d, cmap=cmap, extent=[min_col, max_col + 1, max_row + 1, min_row])
        else:
            plt.imshow(data_2d, cmap=cmap)

        plt.colorbar(label=label)
        if title:
            plt.title(title)
        plt.xlabel('X (grid cells)')
        plt.ylabel('Y (grid cells)')
        if save_path:
            plt.savefig(save_path)
        if self.runtime_plotting:
            plt.show()
        plt.close()

    def adjustOutletToDam(self):
        """Adjust the outlet to a node on the dam if it's in the lake."""
        if not self.grid or not self.grid.has_field('sink_mask'):
            raise ValueError("Cannot adjust outlet: Lake mask is not loaded or invalid.")

        lake_mask = self.grid.at_node['sink_mask'].reshape(self.grid.shape)
        rows, cols = self.grid.shape
        valid_data_2d = self.valid_data_mask.reshape(self.grid.shape)

        boundary_nodes = []
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                node = r * cols + c
                if lake_mask[r, c] == 0 and valid_data_2d[r, c]:
                    neighbors = [
                        (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)
                    ]
                    if any(lake_mask[nr, nc] == 1 for nr, nc in neighbors if 0 <= nr < rows and 0 <= nc < cols):
                        boundary_nodes.append((node, r, c))

        if not boundary_nodes:
            raise ValueError("No suitable boundary nodes found on the dam. Check lake mask or DEM.")

        elevations = [self.grid.at_node['topographic__elevation'][node] for node, _, _ in boundary_nodes]
        min_elev_idx = np.argmin(elevations)
        new_outlet_id, new_row, new_col = boundary_nodes[min_elev_idx]

        self.grid.status_at_node[:] = self.grid.BC_NODE_IS_CLOSED
        self.grid.set_nodata_nodes_to_closed(self.grid.at_node['topographic__elevation'], self.no_data_value)
        self.outlet_id = self.grid.set_watershed_boundary_condition_outlet_id(
            new_outlet_id, self.grid.field_values("topographic__elevation")
        )

        self.logger.log(f"Outlet adjusted to dam: New outlet node ID: {self.outlet_id}")
        self.logger.log(f"New outlet grid position: (row={new_row}, col={new_col})")
        self.logger.log(
            f"New outlet spatial coordinates: ({self.grid.node_x[self.outlet_id]}, {self.grid.node_y[self.outlet_id]})")
        self.logger.log(f"New outlet elevation: {self.grid.at_node['topographic__elevation'][self.outlet_id]}")

    def setUpErosionDepositionModel(self,
                                    m_sp: float = 0.45,
                                    n_sp: float = 1,
                                    K_sp: float = 0.002,
                                    v_s: str = "field_name",
                                    F_f: float = 0.0,
                                    solver: str = 'basic',
                                    runoff_rate: float = None):
        """Set up the erosion and deposition model."""
        if self.grid is None:
            raise ValueError("Grid not created. Call createRasterModelGrid() first.")
        if not self.grid.has_field('water__unit_flux_in') and not runoff_rate:
            raise ValueError(
                "Precipitation rates not set. Call setPrecipitationRates() first or specify runoff_rate keyword argument.")

        self.flow_accumulator = FlowAccumulator(self.grid, depression_finder='DepressionFinderAndRouter',
                                                flow_director='D8', runoff_rate=runoff_rate)
        self.depression_finder = self.flow_accumulator.depression_finder
        self.depression_finder.initialize_optional_output_fields()
        self.sink_filler = SinkFiller(self.grid)
        K = np.ones(self.grid.number_of_nodes) * K_sp
        if self.grid.has_field('sink_mask'):
            K[self.grid.at_node['sink_mask'] == 1] = 0.0
        self.grid.add_field('K', K, at='node', clobber=True)
        if self.grid.has_field('sink_mask'):
            lake_k_values = self.grid.at_node['K'][self.grid.at_node['sink_mask'] == 1]
            if np.any(lake_k_values != 0):
                raise ValueError(f"Error: K values in lake are not all 0. Found values: {np.unique(lake_k_values)}")
            self.logger.log("Confirmed: K is set to 0 in the lake area.")
            self.plotAndSaveFieldData('K', save_path=os.path.join(OUTPUT_DIR, 'K_field.png'))
        self.erosion_deposition_model = ErosionDeposition(self.grid, K='K', m_sp=m_sp, n_sp=n_sp, F_f=F_f, solver=solver)
        self.logger.log("Erosion and deposition model set up successfully.")
        self.logger.log(
            f"Model parameters: m_sp={m_sp}, n_sp={n_sp}, K_sp={K_sp}, v_s={v_s}, F_f={F_f}, solver={solver}")
        self.logger.log(f"Runoff rate: {runoff_rate} m/year")

    def plotFieldStatistics(self, field_name: str, step: int, save_path: str = None):
        """Plot a histogram of the field data, excluding nodata areas."""
        if self.grid is None:
            raise ValueError("Grid not created. Call createRasterModelGrid() first.")
        if field_name not in self.grid.at_node:
            self.logger.log(f"Field {field_name} not available at step {step}. Skipping statistics plot.")
            return

        data = self.grid.at_node[field_name]
        valid_data = data[self.grid.core_nodes]
        valid_data = valid_data[~np.isnan(valid_data)]

        if len(valid_data) == 0:
            self.logger.log(f"No valid data for field {field_name} at step {step}. Skipping statistics plot.")
            return

        plt.figure(figsize=(10, 6))
        plt.hist(valid_data, bins=50, density=True, alpha=0.7, color='blue')
        plt.title(f"Distribution of {field_name} at Step {step}")
        plt.xlabel(field_name)
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            self.logger.log(f"Statistics plot for {field_name} saved to {save_path}")
        if self.runtime_plotting:
            plt.show()
        plt.close()

    def computeFieldMean(self, field_name: str, use_lake_mask: bool = False, exclude_lake: bool = False):
        """Compute the mean of a field, excluding nodata areas, with options for lake masking."""
        if self.grid is None:
            raise ValueError("Grid not created. Call createRasterModelGrid() first.")
        if field_name not in self.grid.at_node:
            return np.nan

        data = self.grid.at_node[field_name]
        if use_lake_mask and self.grid.has_field('sink_mask'):
            mask = self.grid.at_node['sink_mask'] == 1
            valid_data = data[mask]
            self.logger.log(f"Computing mean for {field_name} in lake area:")
            self.logger.log(f"  Number of lake nodes: {np.sum(mask)}")
            self.logger.log(f"  Min value in lake: {np.min(valid_data) if len(valid_data) > 0 else 'N/A'}")
            self.logger.log(f"  Max value in lake: {np.max(valid_data) if len(valid_data) > 0 else 'N/A'}")
            self.logger.log(f"  Mean value in lake: {np.mean(valid_data) if len(valid_data) > 0 else 'N/A'}")
        elif exclude_lake and self.grid.has_field('sink_mask'):
            mask = self.grid.at_node['sink_mask'] == 0
            valid_data = data[self.grid.core_nodes]
            valid_data = valid_data[mask[self.grid.core_nodes]]
            self.logger.log(f"Computing mean for {field_name} excluding lake area:")
            self.logger.log(f"  Number of non-lake core nodes: {np.sum(mask[self.grid.core_nodes])}")
            self.logger.log(f"  Min value outside lake: {np.min(valid_data) if len(valid_data) > 0 else 'N/A'}")
            self.logger.log(f"  Max value outside lake: {np.max(valid_data) if len(valid_data) > 0 else 'N/A'}")
            self.logger.log(f"  Mean value outside lake: {np.mean(valid_data) if len(valid_data) > 0 else 'N/A'}")
        else:
            valid_data = data[self.grid.core_nodes]
        valid_data = valid_data[~np.isnan(valid_data)]

        if len(valid_data) == 0:
            return np.nan
        return np.mean(valid_data)

    def computeTotalWaterInput(self):
        """Compute the total water input (runoff rate × area) over all core nodes."""
        if self.grid is None or 'water__unit_flux_in' not in self.grid.at_node:
            return np.nan
        runoff = self.grid.at_node['water__unit_flux_in']
        cell_area = self.grid.dx * self.grid.dy
        valid_runoff = runoff[self.grid.core_nodes]
        valid_runoff = valid_runoff[~np.isnan(valid_runoff)]
        total_water = np.sum(valid_runoff) * cell_area
        return total_water

    def plotDrainageNetwork(self, step: int, save_path: str = None):
        """Plot the drainage network based on surface_water__discharge."""
        if self.grid is None:
            raise ValueError("Grid not created. Call createRasterModelGrid() first.")
        if 'surface_water__discharge' not in self.grid.at_node:
            self.logger.log(f"Field surface_water__discharge not available at step {step}. Skipping drainage network plot.")
            return

        plt.figure(figsize=(10, 10))
        discharge = self.grid.at_node['surface_water__discharge'].reshape(self.grid.shape)
        discharge[self.nodata_mask] = np.nan
        log_discharge = np.log10(discharge + 1e-6)
        plt.imshow(log_discharge, cmap='Blues')
        plt.colorbar(label='Log10(Discharge) (m³/year)')
        if self.grid.has_field('sink_mask'):
            plt.contour(self.grid.at_node["sink_mask"].reshape(self.grid.shape), levels=[0.5], colors='red', linestyles='--', linewidths=2)
        plt.title(f"Drainage Network (Log Discharge) at Step {step}")
        plt.xlabel('X (grid cells)')
        plt.ylabel('Y (grid cells)')
        if save_path:
            plt.savefig(save_path)
            self.logger.log(f"Drainage network plot saved to {save_path}")
        if self.runtime_plotting:
            plt.show()
        plt.close()

    def plotTrends(self, field_name: str, years: list, values: list, ylabel: str):
        """Plot the trend of a field's mean value over time."""
        plt.figure(figsize=(10, 6))
        plt.plot(years, values, marker='o', color='blue')
        plt.title(f"Trend of Mean {field_name} Over Time")
        plt.xlabel('Simulation Year')
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        save_path = os.path.join(TRENDS_OUTPUT_DIR, f'{field_name}_trend.png')
        plt.savefig(save_path)
        self.logger.log(f"Trend plot for {field_name} saved to {save_path}")
        if self.runtime_plotting:
            plt.show()
        plt.close()

    def plotCombinedTrends(self, field_name1: str, field_name2: str, years: list, values1: list, values2: list, ylabel: str, title: str, color1: str = 'blue', color2: str = 'red'):
        """Plot two trends on the same graph with different colors."""
        plt.figure(figsize=(10, 6))
        plt.plot(years, values1, marker='o', color=color1, label=field_name1)
        plt.plot(years, values2, marker='o', color=color2, label=field_name2)
        plt.title(title)
        plt.xlabel('Simulation Year')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_path = os.path.join(TRENDS_OUTPUT_DIR, f'{field_name1}_vs_{field_name2}_trend.png')
        plt.savefig(save_path)
        self.logger.log(f"Combined trend plot for {field_name1} and {field_name2} saved to {save_path}")
        if self.runtime_plotting:
            plt.show()
        plt.close()

    def calculate_lake_fill_time(self):
        """Calculate the time required to completely fill the lake with sediment."""
        if not self.grid or not self.grid.has_field('sink_mask'):
            raise ValueError("Grid or sink_mask not initialized. Run createRasterModelGrid() and setUpErosionDepositionModel() first.")

        lake_nodes = np.sum(self.grid.at_node['sink_mask'] == 1)
        cell_area = self.grid.dx * self.grid.dy
        lake_area = lake_nodes * cell_area

        initial_average_depth = 100
        initial_volume = lake_area * initial_average_depth

        elevation_diff = self.grid.at_node['topographic__elevation'] - self.initial_topography
        elevation_diff_lake = elevation_diff.copy()
        elevation_diff_lake[self.grid.at_node['sink_mask'] == 0] = np.nan
        mean_elevation_increase = np.nanmean(elevation_diff_lake) if np.any(~np.isnan(elevation_diff_lake)) else 0

        remaining_average_depth = initial_average_depth - mean_elevation_increase
        present_day_volume = lake_area * max(0, remaining_average_depth)

        self.logger.log(f"Number of lake nodes: {lake_nodes}")
        self.logger.log(f"Lake area: {lake_area} m²")
        self.logger.log(f"Initial average depth: {initial_average_depth} m")
        self.logger.log(f"Initial volume: {initial_volume} m³")
        self.logger.log(f"Mean elevation increase after 1000 years: {mean_elevation_increase} m")
        self.logger.log(f"Present-day volume: {present_day_volume} m³")

        sediment_volume = lake_area * mean_elevation_increase
        self.logger.log(f"Sediment volume deposited in 1000 years: {sediment_volume} m³")

        sedimentation_rate = sediment_volume / 1000
        self.logger.log(f"Mean sedimentation rate: {sedimentation_rate} m³/year")

        time_to_fill = present_day_volume / sedimentation_rate if sedimentation_rate > 0 else float('inf')
        total_time_mean = 1000 + time_to_fill
        self.logger.log(f"Time to fill lake (using mean rate): {total_time_mean:.0f} years")

        lake_mask = self.grid.at_node['sink_mask'] == 1
        min_elevation_increase = np.nanmin(elevation_diff_lake[lake_mask]) if np.any(lake_mask) else 0
        max_initial_depth = 200
        remaining_depth_min = max_initial_depth - min_elevation_increase
        sediment_volume_min_per_node = min_elevation_increase * cell_area
        sediment_rate_min_per_node = sediment_volume_min_per_node / 1000
        remaining_volume_per_node = remaining_depth_min * cell_area
        time_to_fill_min_per_node = remaining_volume_per_node / sediment_rate_min_per_node if sediment_rate_min_per_node > 0 else float('inf')
        total_time_min = 1000 + time_to_fill_min_per_node
        self.logger.log(f"Minimum elevation increase: {min_elevation_increase} m")
        self.logger.log(f"Maximum initial depth: {max_initial_depth} m")
        self.logger.log(f"Remaining depth in deepest areas: {remaining_depth_min} m")
        self.logger.log(f"Minimum sedimentation rate per node: {sediment_rate_min_per_node} m³/year")
        self.logger.log(f"Time to fill lake (using minimum rate in deepest areas): {total_time_min:.0f} years")

        return total_time_mean, total_time_min

    def calculateLakeFillTimeWithDepth(self):
        """Calculate the time required to completely fill the lake using depth map and sediment deposition."""
        if not self.grid or not self.grid.has_field('sink_mask'):
            raise ValueError("Grid or sink_mask not initialized. Run createRasterModelGrid() first.")

        elevation_diff = self.grid.at_node['topographic__elevation'] - self.initial_topography
        elevation_diff_2d = elevation_diff.reshape(self.grid.shape)
        lake_mask_2d = self.grid.at_node['sink_mask'].reshape(self.grid.shape)
        elevation_diff_lake = elevation_diff_2d * lake_mask_2d
        elevation_diff_lake[self.nodata_mask] = np.nan
        cell_area = self.grid.dx * self.grid.dy
        sediment_volume_per_pixel = elevation_diff_lake * cell_area
        V_sediment = np.nansum(sediment_volume_per_pixel) * 1e-9
        self.logger.log(f"Calculated sediment volume deposited (V_sed): {V_sediment} km³")

        if self.depth_map is None:
            raise ValueError("Depth map not loaded. Please provide a valid path_to_depth_map when initializing the simulator.")
        depth_map_2d = self.depth_map.reshape(self.grid.shape)
        depth_map_lake = depth_map_2d * lake_mask_2d
        depth_map_lake[self.nodata_mask] = np.nan
        lake_volume_per_pixel = depth_map_lake * cell_area
        V_lake = np.nansum(lake_volume_per_pixel) * 1e-9
        self.logger.log(f"Calculated present-day lake volume (V_lake): {V_lake} km³")

        T_model = self.model_runtime if self.model_runtime is not None else 100
        sedimentation_rate = V_sediment / T_model if T_model > 0 else 0
        self.logger.log(f"Sedimentation rate: {sedimentation_rate} km³/year")
        T_fill = V_lake / sedimentation_rate if sedimentation_rate > 0 else float('inf')
        total_years = T_model + T_fill
        self.logger.log(f"Total time to completely fill the lake: {total_years:.0f} years")
        return total_years

    def runSimulation(self, years: int = 100, dt: float = 1, uplift_rate: float = 0.002):
        """Run the simulation for a specified number of years."""
        if self.flow_accumulator is None or self.erosion_deposition_model is None:
            raise ValueError("Erosion and deposition model not set up. Call setUpErosionDepositionModel() first.")

        self.logger.log(f"Running simulation for {years} years with dt={dt} years.")
        self.model_runtime = years
        n_steps = int(years / dt)

        stats_fields = [
            'flow__link_to_receiver_node',
            'sediment__flux',
            'sediment__influx',
            'sediment__outflux',
            'surface_water__discharge',
            'topographic__elevation',
            'topographic__steepest_slope'
        ]

        print("Generating initial statistics and maps at step 0...")
        self.flow_accumulator.run_one_step()
        self.erosion_deposition_model.run_one_step(dt=dt)
        if self.grid.has_field('sink_mask'):
            self.grid.at_node['sediment__outflux'][self.grid.at_node['sink_mask'] == 1] = 0.0
        self.logger.log(f"Step 0: Number of core nodes: {len(self.grid.core_nodes)}")
        total_water = self.computeTotalWaterInput()
        self.logger.log(f"Step 0: Total water input: {total_water} m³/year")
        for field in stats_fields:
            if field in self.grid.at_node:
                self.plotFieldStatistics(field, step=0, save_path=os.path.join(STATS_OUTPUT_DIR, f'{field}_stats_step_0.png'))
                self.plotAndSaveFieldData(field, save_path=os.path.join(GRID_OUTPUT_DIR, f'{field}_step_0.png'), overlay_sink_mask=True)
                self.trends[field].append(self.computeFieldMean(field))
                if field == 'sediment__flux' and self.grid.has_field('sink_mask'):
                    self.trends['sediment__flux_lake'].append(self.computeFieldMean(field, use_lake_mask=True))
                if field in ['sediment__influx', 'sediment__outflux'] and self.grid.has_field('sink_mask'):
                    self.trends[f'{field}_lake'].append(self.computeFieldMean(field, use_lake_mask=True))
                if field == 'surface_water__discharge' and self.grid.has_field('sink_mask'):
                    self.trends['surface_water__discharge_non_lake'].append(self.computeFieldMean(field, exclude_lake=True))
                if field == 'surface_water__discharge' and self.outlet_id is not None:
                    outlet_discharge = self.grid.at_node['surface_water__discharge'][self.outlet_id]
                    self.trends['surface_water__discharge_outlet'].append(outlet_discharge)
                    self.logger.log(f"Step 0: Discharge at outlet: {outlet_discharge} m³/year")
        self.initial_mean_elevation = self.computeFieldMean('topographic__elevation')
        self.trends['topography_change'].append(0.0)
        if self.grid.has_field('sink_mask'):
            elevation_diff = self.grid.at_node['topographic__elevation'] - self.initial_topography
            elevation_diff_lake = elevation_diff.copy()
            elevation_diff_lake[self.grid.at_node['sink_mask'] == 0] = np.nan
            mean_diff_lake = np.nanmean(elevation_diff_lake)
            self.trends['elevation_difference_lake'].append(mean_diff_lake)
        else:
            self.trends['elevation_difference_lake'].append(np.nan)
        self.trend_years.append(0)

        self.plotDrainageNetwork(step=0, save_path=os.path.join(GRID_OUTPUT_DIR, 'drainage_network_step_0.png'))

        self.grid.at_node['topographic__elevation'][:] = self.initial_topography[:]

        interval = max(1, int(years / 10))
        for i in range(n_steps):
            current_year = int((i + 1) * dt)
            print(f"Running simulation step {i+1}/{n_steps} (Year {current_year})...")
            self.flow_accumulator.run_one_step()
            self.erosion_deposition_model.run_one_step(dt=dt)
            if self.grid.has_field('sink_mask'):
                self.grid.at_node['sediment__outflux'][self.grid.at_node['sink_mask'] == 1] = 0.0
            uplift_mask = self.valid_data_mask.copy().reshape(self.grid.shape)
            if self.grid.has_field('sink_mask'):
                sink_mask_2d = self.grid.at_node['sink_mask'].reshape(self.grid.shape)
                uplift_mask[sink_mask_2d == 1] = False
            elevation_2d = self.grid.at_node['topographic__elevation'].reshape(self.grid.shape)
            elevation_2d[uplift_mask] += uplift_rate * dt
            self.grid.at_node['topographic__elevation'][:] = elevation_2d.flatten()
            print(f"Step {i+1}/{n_steps} completed.")

            if (i + 1) % interval == 0 or i == n_steps - 1:
                self.logger.log("+-" * 50)
                self.logger.log(f"Simulation step {i+1}/{n_steps} completed.")
                print(f"Simulation step {i+1}/{n_steps} completed.")
                self.logger.log(f"Step {i+1}: Number of core nodes: {len(self.grid.core_nodes)}")
                total_water = self.computeTotalWaterInput()
                self.logger.log(f"Step {i+1}: Total water input: {total_water} m³/year")

                for field in stats_fields:
                    if field in self.grid.at_node:
                        self.trends[field].append(self.computeFieldMean(field))
                        if field == 'sediment__flux' and self.grid.has_field('sink_mask'):
                            self.trends['sediment__flux_lake'].append(self.computeFieldMean(field, use_lake_mask=True))
                        if field in ['sediment__influx', 'sediment__outflux'] and self.grid.has_field('sink_mask'):
                            self.trends[f'{field}_lake'].append(self.computeFieldMean(field, use_lake_mask=True))
                        if field == 'surface_water__discharge' and self.grid.has_field('sink_mask'):
                            self.trends['surface_water__discharge_non_lake'].append(self.computeFieldMean(field, exclude_lake=True))
                        if field == 'surface_water__discharge' and self.outlet_id is not None:
                            outlet_discharge = self.grid.at_node['surface_water__discharge'][self.outlet_id]
                            self.trends['surface_water__discharge_outlet'].append(outlet_discharge)
                            self.logger.log(f"Step {i+1}: Discharge at outlet: {outlet_discharge} m³/year")
                current_mean_elevation = self.computeFieldMean('topographic__elevation')
                self.trends['topography_change'].append(current_mean_elevation - self.initial_mean_elevation)
                if self.grid.has_field('sink_mask'):
                    elevation_diff = self.grid.at_node['topographic__elevation'] - self.initial_topography
                    elevation_diff_lake = elevation_diff.copy()
                    elevation_diff_lake[self.grid.at_node['sink_mask'] == 0] = np.nan
                    mean_diff_lake = np.nanmean(elevation_diff_lake)
                    self.trends['elevation_difference_lake'].append(mean_diff_lake)
                else:
                    self.trends['elevation_difference_lake'].append(np.nan)
                self.trend_years.append(current_year)

                grid_fields = [f.split(":")[1] for f in self.grid.fields()]
                print(f"Grid fields: {grid_fields}")
                for field in grid_fields:
                    if field in self.grid.at_node:
                        self.logFieldStats(field)
                        self.plotAndSaveFieldData(field, save_path=os.path.join(GRID_OUTPUT_DIR, f'{field}_step_{i+1}.png'), overlay_sink_mask=True)

                flood_status = self.depression_finder.flood_status.reshape(self.grid.shape)
                self.plotAndSaveDataArray(flood_status,
                                          title=f"Flood Status Step {i+1}",
                                          cmap='RdBu',
                                          label='Flood Status',
                                          save_path=os.path.join(DF_OUTPUT_DIR, f'flood_status_step_{i+1}.png'))

                lake_map = self.depression_finder.lake_map.reshape(self.grid.shape)
                self.plotAndSaveDataArray(lake_map,
                                          title=f"Lake Map Step {i+1}",
                                          cmap='RdBu',
                                          label='Lake Map',
                                          save_path=os.path.join(DF_OUTPUT_DIR, f'lake_map_step_{i+1}.png'))

                depression_depth = self.depression_finder.depression_depth.reshape(self.grid.shape)
                self.plotAndSaveDataArray(depression_depth,
                                          title=f"Depression Depth Step {i+1}",
                                          cmap='RdBu',
                                          label='Depression Depth (m)',
                                          save_path=os.path.join(DF_OUTPUT_DIR, f'depression_depth_step_{i+1}.png'))

                depression_outlet_map = self.depression_finder.depression_outlet_map.reshape(self.grid.shape)
                self.plotAndSaveDataArray(depression_outlet_map,
                                          title=f"Depression Outlet Map Step {i+1}",
                                          cmap='RdBu',
                                          label='Depression Outlet Map',
                                          save_path=os.path.join(DF_OUTPUT_DIR, f'depression_outlet_map_step_{i+1}.png'))

                elevation_diff = self.grid.at_node['topographic__elevation'] - self.initial_topography
                self.plotAndSaveDataArray(elevation_diff,
                                          title=f"Topography Change Step {i+1}",
                                          cmap='RdBu_r',
                                          label='Elevation Change (m)',
                                          save_path=os.path.join(OUTPUT_DIR, f'topography_change_step_{i+1}.png'))

                self.plotDrainageNetwork(step=i+1, save_path=os.path.join(GRID_OUTPUT_DIR, f'drainage_network_step_{i+1}.png'))

                if i == n_steps - 1:
                    for field in stats_fields:
                        if field in self.grid.at_node:
                            self.plotFieldStatistics(field, step=i+1, save_path=os.path.join(STATS_OUTPUT_DIR, f'{field}_stats_step_{i+1}.png'))

                    elevation_diff = self.grid.at_node['topographic__elevation'] - self.initial_topography
                    if self.grid.has_field('sink_mask'):
                        elevation_diff_lake = elevation_diff.copy()
                        elevation_diff_lake[self.grid.at_node['sink_mask'] == 0] = np.nan
                        plt.figure(figsize=(10, 6))
                        valid_diff = elevation_diff_lake[~np.isnan(elevation_diff_lake)]
                        if len(valid_diff) > 0:
                            plt.hist(valid_diff, bins=50, density=True, alpha=0.7, color='blue')
                            plt.title(f"Distribution of Elevation Difference (Lake Only) at Step {i+1}")
                            plt.xlabel('Elevation Change (m)')
                            plt.ylabel('Density')
                            plt.grid(True, alpha=0.3)
                            save_path = os.path.join(STATS_OUTPUT_DIR, f'elevation_difference_lake_stats_step_{i+1}.png')
                            plt.savefig(save_path)
                            self.logger.log(f"Statistics plot for Elevation Difference (Lake Only) saved to {save_path}")
                            plt.close()
                        else:
                            self.logger.log(f"No valid data for Elevation Difference (Lake Only) at step {i+1}. Skipping statistics plot.")

                    plt.figure(figsize=(10, 6))
                    valid_diff = elevation_diff[~np.isnan(elevation_diff)]
                    if len(valid_diff) > 0:
                        plt.hist(valid_diff, bins=50, density=True, alpha=0.7, color='blue')
                        plt.title(f"Distribution of Topography Change at Step {i+1}")
                        plt.xlabel('Elevation Change (m)')
                        plt.ylabel('Density')
                        plt.grid(True, alpha=0.3)
                        save_path = os.path.join(STATS_OUTPUT_DIR, f'topography_change_stats_step_{i+1}.png')
                        plt.savefig(save_path)
                        self.logger.log(f"Statistics plot for Topography Change saved to {save_path}")
                        plt.close()
                    else:
                        self.logger.log(f"No valid data for Topography Change at step {i+1}. Skipping statistics plot.")

        elevation_diff = self.grid.at_node['topographic__elevation'] - self.initial_topography
        if self.grid.has_field('sink_mask'):
            elevation_diff_lake = elevation_diff.copy()
            elevation_diff_lake[self.grid.at_node['sink_mask'] == 0] = np.nan
            self.plotAndSaveDataArray(elevation_diff_lake,
                                      title="Elevation Difference (Lake Only)",
                                      save_path=os.path.join(OUTPUT_DIR, 'elevation_difference_lake.png'),
                                      cmap='RdBu_r',
                                      label='Elevation Change (m)',
                                      crop_to_lake=True)

        print("Plotting trends over time...")
        self.plotTrends('flow__link_to_receiver_node', self.trend_years, self.trends['flow__link_to_receiver_node'], 'Mean Link ID')
        self.plotTrends('sediment__flux', self.trend_years, self.trends['sediment__flux'], 'Mean Sediment Flux (m³/year)')
        self.plotTrends('sediment__flux_lake', self.trend_years, self.trends['sediment__flux_lake'], 'Mean Sediment Flux in Lake (m³/year)')
        self.plotTrends('sediment__influx', self.trend_years, self.trends['sediment__influx'], 'Mean Sediment Influx (m³/year)')
        self.plotTrends('sediment__outflux', self.trend_years, self.trends['sediment__outflux'], 'Mean Sediment Outflux (m³/year)')
        self.plotTrends('surface_water__discharge', self.trend_years, self.trends['surface_water__discharge'], 'Mean Discharge (m³/year)')
        self.plotTrends('surface_water__discharge_non_lake', self.trend_years, self.trends['surface_water__discharge_non_lake'], 'Mean Discharge Outside Lake (m³/year)')
        self.plotTrends('surface_water__discharge_outlet', self.trend_years, self.trends['surface_water__discharge_outlet'], 'Discharge at Outlet (m³/year)')
        self.plotTrends('topographic__elevation', self.trend_years, self.trends['topographic__elevation'], 'Mean Elevation (m)')
        self.plotTrends('topographic__steepest_slope', self.trend_years, self.trends['topographic__steepest_slope'], 'Mean Slope (dimensionless)')
        self.plotTrends('elevation_difference_lake', self.trend_years, self.trends['elevation_difference_lake'], 'Mean Elevation Change in Lake (m)')
        self.plotTrends('topography_change', self.trend_years, self.trends['topography_change'], 'Mean Topography Change (m)')
        self.plotCombinedTrends(
            'sediment__influx',
            'sediment__outflux',
            self.trend_years,
            self.trends['sediment__influx'],
            self.trends['sediment__outflux'],
            'Mean Sediment Flux (m³/year)',
            'Trend of Mean Sediment Influx and Outflux Over Time',
            color1='blue',
            color2='red'
        )
        if self.grid.has_field('sink_mask'):
            self.plotCombinedTrends(
                'sediment__influx_lake',
                'sediment__outflux_lake',
                self.trend_years,
                self.trends['sediment__influx_lake'],
                self.trends['sediment__outflux_lake'],
                'Mean Sediment Flux in Lake (m³/year)',
                'Trend of Mean Sediment Influx and Outflux in Lake Over Time',
                color1='blue',
                color2='red'
            )

        if self.grid.has_field('sink_mask') and years >= 1000:
            total_time_mean, total_time_min = self.calculate_lake_fill_time()
            self.logger.log(f"Estimated total time to fill lake (mean rate): {total_time_mean:.0f} years")
            self.logger.log(f"Estimated total time to fill lake (minimum rate in deepest areas): {total_time_min:.0f} years")

        if self.grid.has_field('sink_mask') and self.depth_map is not None:
            total_years = self.calculateLakeFillTimeWithDepth()
            self.logger.log(f"Estimated total time to fill lake using depth map: {total_years:.0f} years")
            print(f"Estimated total time to fill lake using depth map: {total_years:.0f} years")

    def plotFieldData(self, field_name: str):
        """Plot the field data."""
        if self.grid is None:
            raise ValueError("Grid not created. Call createRasterModelGrid() first.")
        if field_name not in self.grid.at_node:
            raise ValueError(f"Field {field_name} not found in the grid.")
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid.at_node[field_name].reshape(self.grid.shape), cmap='terrain')
        plt.colorbar(label=field_name)
        plt.show()

    def plotAndSaveDataArray(self, data_array: np.ndarray, title: str = None, save_path: str = None,
                             cmap: str = 'viridis', label: str = 'Value', crop_to_lake: bool = False):
        """Plot and save a data array."""
        if not title:
            title = "output_result"
        if not save_path:
            save_path = os.path.join(OUTPUT_DIR, f'{title}.png')
        self.plotDataArray(data_array, title=title, save_path=save_path, cmap=cmap, label=label, crop_to_lake=crop_to_lake)

    def plotAndSaveFieldData(self, field_name: str, save_path: str = None, overlay_sink_mask: bool = False):
        """Plot and save field data."""
        if self.grid is None:
            raise ValueError("Grid not created. Call createRasterModelGrid() first.")
        if field_name not in self.grid.at_node:
            raise ValueError(f"Field {field_name} not found in the grid.")
        if not save_path:
            save_path = os.path.join(OUTPUT_DIR, f'{field_name}.png')

        if field_name in ['sediment__flux', 'sediment__influx', 'sediment__outflux']:
            cmap = 'RdBu_r'
        elif field_name == 'topographic__elevation':
            cmap = 'terrain'
        else:
            cmap = 'RdBu'
        plt.figure(figsize=(10, 10))
        data_2d = self.grid.at_node[field_name].reshape(self.grid.shape)
        if np.issubdtype(data_2d.dtype, np.integer):
            data_2d = data_2d.astype(np.float64)
        data_2d[self.nodata_mask] = np.nan
        plt.imshow(data_2d, cmap=cmap)
        plt.colorbar(label=field_name)
        if overlay_sink_mask and self.grid.has_field('sink_mask'):
            plt.contour(self.grid.at_node["sink_mask"].reshape(self.grid.shape), levels=[0.5], colors='red', linestyles='--', linewidths=2)
        plt.title(field_name)
        plt.xlabel('X (grid cells)')
        plt.ylabel('Y (grid cells)')
        plt.savefig(save_path)
        plt.close()

    def printFieldStats(self, field_name: str):
        """Print statistics about a field data."""
        if self.grid is None:
            raise ValueError("Grid not created. Call createRasterModelGrid() first.")
        if field_name not in self.grid.at_node:
            raise ValueError(f"Field {field_name} not found in the grid.")
        print(f"Field {field_name}:")
        print(f"  Shape: {self.grid.at_node[field_name].shape}")
        print(f"  Data type: {self.grid.at_node[field_name].dtype}")
        print(f"  Units: {self.grid.field_units(field_name)}")
        print(f"  Min: {self.grid.at_node[field_name].min()}")
        print(f"  Max: {self.grid.at_node[field_name].max()}")
        print(f"  Mean: {self.grid.at_node[field_name].mean()}")

    def logFieldStats(self, field_name: str):
        """Log statistics about a field data."""
        if self.grid is None:
            raise ValueError("Grid not created. Call createRasterModelGrid() first.")
        if field_name not in self.grid.at_node:
            raise ValueError(f"Field {field_name} not found in the grid.")
        self.logger.log(f"Field:\t {field_name}:")
        self.logger.log(f"Shape:\t {self.grid.at_node[field_name].shape}")
        self.logger.log(f"Data type:\t {self.grid.at_node[field_name].dtype}")
        self.logger.log(f"Units:\t {self.grid.field_units(field_name)}")
        self.logger.log(f"Min:\t {self.grid.at_node[field_name][self.grid.core_nodes].min()}")
        self.logger.log(f"Max:\t {self.grid.at_node[field_name][self.grid.core_nodes].max()}")
        self.logger.log(f"Mean:\t {self.grid.at_node[field_name][self.grid.core_nodes].mean()}")

    def setNoData(self, nodata_value: float):
        """Set the nodata value for the DEM."""
        self.no_data_value = nodata_value
        self.logger.log(f"Set no-data value to {self.no_data_value}.")

    def getNoData(self):
        """Get the nodata value."""
        return self.no_data_value

    def getGrid(self):
        """Get the grid."""
        return self.grid

    def setRunTimePlotting(self, runtime_plotting: bool):
        """Set the run time plotting flag."""
        self.runtime_plotting = runtime_plotting

    def saveStepResults(self, save_step_results: bool):
        """Set the save step results flag."""
        self.save_step_results = save_step_results
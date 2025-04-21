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

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(GRID_OUTPUT_DIR):
    os.makedirs(GRID_OUTPUT_DIR)
if not os.path.exists(DF_OUTPUT_DIR):
    os.makedirs(DF_OUTPUT_DIR)



class SourceToSinkSimulator:
    def __init__(self, path_to_topography: str, path_to_precipitation: str = None, path_to_watershed_mask:str=None, path_to_sink_mask:str=None, runtime_plotting: bool = False):
        self.path_to_topography = path_to_topography
        self.path_to_precipitation = path_to_precipitation
        self.path_to_watershed_mask = path_to_watershed_mask
        self.path_to_sink_mask = path_to_sink_mask
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

        self.runtime_plotting = runtime_plotting
        self.save_step_results = True
        self.outlet_id = None
        self.logger = ProcessLogger(os.path.join(OUTPUT_DIR, 'process.log'))
        self.logger.log("SourceToSinkSimulator initialized.")

    def loadRaster(self, raster_path: str, is_mask: bool = False, is_dem: bool = False):
        """Load a raster file."""
        try:
            with rasterio.open(raster_path) as src:
                if src.crs.is_geographic:
                    dst_path = raster_path.replace('.tif', '_utm.tif')
                    self.reprojectToUtm(src, dst_path)
                    src = rasterio.open(dst_path)
                    print(f"Reprojected raster to UTM and saved as {dst_path}.")
                if  self.grid and src.shape != self.grid.shape:
                    print(f"grid shape: {self.grid.shape}, raster shape: {src.shape}")
                    print(f"Warning: The resolution of the {raster_path} raster does not match the DEM resolution.")
                    print("Resampling the raster to match the DEM resolution.")
                    path_to_raster = resampleDEM(raster_path,
                                                            up_scale_factor=src.res[0] / self.model_resolution)
                    src = rasterio.open(path_to_raster)

                if  is_mask:
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

        # Load the DEM raster
        elevation_data = self.loadRaster(self.path_to_topography, is_dem=True)
        if elevation_data is None:
            raise ValueError("Elevation data not found or invalid. Please check the path.")
        self.shape = elevation_data.shape

        # Mask out pixels outside the watershed area
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

        # Create a raster model grid
        self.grid = RasterModelGrid(elevation_data.shape, xy_spacing=(self.model_x_spacing, self.model_y_spacing))
        self.grid.add_field('topographic__elevation', elevation_data, at='node', units='m')

        # Add the watershed mask as a field in the grid
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
                # Convert precipitation from mm/year to m/year
                precipitation = precipitation_data / 1000.0
                precipitation[self.nodata_mask] = self.no_data_value    
                # Add precipitation as a field in the grid
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
        else:
            warnings.warn("No sink mask provided. Proceeding without it. Some plots may not be available.")

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

            # Compute min and max elevations in the watershed (core nodes only)
            elevation = self.grid.at_node['topographic__elevation']
            core_elevations = elevation[self.grid.core_nodes]
            min_elev = np.min(core_elevations)
            min_node = self.grid.core_nodes[np.argmin(core_elevations)]
            max_elev = np.max(core_elevations)
            max_node = self.grid.core_nodes[np.argmax(core_elevations)]

            # Print to console
            print(f"Minimum elevation in watershed: {min_elev} m at node {min_node}")
            print(f"Maximum elevation in watershed: {max_elev} m at node {max_node}")
            print(
                f"Outlet node: {self.outlet_id}, elevation: {self.grid.at_node['topographic__elevation'][self.outlet_id]} m")

            # Log additional details
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
            # Find the minimum elevation in the core nodes
            # and set the outlet to that node
            # This is a fallback in case the above method fails
            # due to multiple minimum elevations
            # or other issues.
            # Find the minimum elevation in the core nodes
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
                      label: str = 'Value'):
        """Plot a data array, optionally saving to a file."""
        if self.grid is None:
            raise ValueError("Grid not created. Call createRasterModelGrid() first.")
        plt.figure(figsize=(10, 10))
        data_2d = data_array.reshape(self.grid.shape) if data_array.ndim == 1 else data_array
        
        if not "Watershed area" in title:
            if data_array.dtype == np.int32:
                data_2d = data_2d.astype(np.float64)
            data_2d[self.nodata_mask] = np.nan
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
        self.erosion_deposition_model = ErosionDeposition(self.grid, K=K_sp, m_sp=m_sp, n_sp=n_sp, F_f=F_f,
                                                          solver=solver)
        self.logger.log("Erosion and deposition model set up successfully.")
        self.logger.log(
            f"Model parameters: m_sp={m_sp}, n_sp={n_sp}, K_sp={K_sp}, v_s={v_s}, F_f={F_f}, solver={solver}")
        self.logger.log(f"Runoff rate: {runoff_rate} m/s")

    def runSimulation(self, years: int = 100, dt: float = 1, uplift_rate: float = 0.002):
        """Run the simulation for a specified number of years."""
        if self.flow_accumulator is None or self.erosion_deposition_model is None:
            raise ValueError("Erosion and deposition model not set up. Call setUpErosionDepositionModel() first.")
        
        self.logger.log(f"Running simulation for {years} years with dt={dt} seconds.")
        self.model_runtime = years
        n_steps = int(years / dt)
        for i in range(n_steps):
            print(f"Running simulation step {i+1}/{n_steps}...")
            self.flow_accumulator.run_one_step()
            #self.sink_filler.run_one_step()
            self.erosion_deposition_model.run_one_step(dt=dt)
            self.grid.at_node['topographic__elevation'].reshape(self.grid.shape)[
                self.valid_data_mask] += uplift_rate * dt
            print(f"Step {i+1}/{n_steps} completed.")
            if i % (years / 10) == 0 or i == n_steps - 1:
                self.logger.log("+-" * 50)
                self.logger.log(f"Simulation step {i+1}/{n_steps} completed.")
                print(f"Simulation step {i+1}/{n_steps} completed.")

                #plot all the fields in the grid. Th final results will be plotted after last step. 
                grid_fields = [i.split(":")[1] for i in self.grid.fields()]
                print(f"Grid fields: {grid_fields}")
                for field in grid_fields:
                    if field in self.grid.at_node:
                        self.logFieldStats(field)
                        self.plotAndSaveFieldData(field, save_path=os.path.join(GRID_OUTPUT_DIR, f'{field}_step_{i+1}.png'))

                # Plot flood status of the nodes
                flood_status = self.depression_finder.flood_status.reshape(self.grid.shape)
                self.plotAndSaveDataArray(flood_status,
                                          title=f"Flood Status Step {i+1}",
                                          cmap='RdBu',
                                          label='Flood Status',
                                          save_path=os.path.join(DF_OUTPUT_DIR, f'flood_status_step_{i+1}.png'))

                # Plot lake map
                lake_map = self.depression_finder.lake_map.reshape(self.grid.shape)
                self.plotAndSaveDataArray(lake_map,
                                          title=f"Lake Map Step {i+1}",
                                          cmap='RdBu',
                                          label='Lake Map',
                                          save_path=os.path.join(DF_OUTPUT_DIR, f'lake_map_step_{i+1}.png'))
                
                #Plot depression depths
                depression_depth = self.depression_finder.depression_depth.reshape(self.grid.shape)
                self.plotAndSaveDataArray(depression_depth,
                                          title=f"Depression Depth Step {i+1}",
                                          cmap='RdBu',
                                          label='Depression Depth (m)',
                                          save_path=os.path.join(DF_OUTPUT_DIR, f'depression_depth_step_{i+1}.png'))
                
                #Plot depression outlet map
                depression_outlet_map = self.depression_finder.depression_outlet_map.reshape(self.grid.shape)
                self.plotAndSaveDataArray(depression_outlet_map,
                                            title=f"Depression Outlet Map Step {i+1}",
                                            cmap='RdBu',
                                            label='Depression Outlet Map',
                                            save_path=os.path.join(DF_OUTPUT_DIR, f'depression_outlet_map_step_{i+1}.png'))
                
                elevation_diff = self.grid.at_node['topographic__elevation'] - self.initial_topography
                self.plotAndSaveDataArray(elevation_diff,
                                          title=f"Topography Change Step {i+1}",
                                          cmap='RdBu',
                                          label='Elevation Change (m)',
                                          save_path=os.path.join(OUTPUT_DIR, f'topography_change_step_{i+1}.png'))
        
        elevation_diff = self.grid.at_node['topographic__elevation'] - self.initial_topography
        if self.grid.has_field('sink_mask'):
            elevation_diff[self.grid.at_node['sink_mask'] == 0] = np.nan
            self.plotAndSaveDataArray(elevation_diff.reshape(self.grid.shape),
                                    title="Elevation Difference (Lake Only)",
                                    save_path=os.path.join(OUTPUT_DIR, 'elevation_difference_lake.png'),
                                    cmap='RdBu',
                                    label='Elevation Change (m)')

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
                             cmap: str = 'viridis', label: str = 'Value'):
        """Plot and save a data array."""
        if not title:
            title = "output_result"
        if not save_path:
            save_path = os.path.join(OUTPUT_DIR, f'{title}.png')
        self.plotDataArray(data_array, title=title, save_path=save_path, cmap=cmap, label=label)

    def plotAndSaveFieldData(self, field_name: str, save_path: str = None):
        """Plot and save field data."""
        if self.grid is None:
            raise ValueError("Grid not created. Call createRasterModelGrid() first.")
        if field_name not in self.grid.at_node:
            raise ValueError(f"Field {field_name} not found in the grid.")
        if not save_path:
            save_path = os.path.join(OUTPUT_DIR, f'{field_name}.png')
        
        if field_name == 'topographic__elevation':
            cmap = 'terrain'
        else:
            cmap = 'RdBu'
        plt.figure(figsize=(10, 10))
        data_2d = self.grid.at_node[field_name].reshape(self.grid.shape)
        if self.grid.at_node[field_name].dtype == np.int32:
            data_2d = data_2d.astype(np.float64)
        data_2d[self.nodata_mask] = np.nan
        plt.imshow(data_2d, cmap=cmap)
        plt.colorbar(label=field_name)
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
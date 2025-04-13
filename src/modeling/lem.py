from landlab import RasterModelGrid
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

from landlab.components import FlowAccumulator, ErosionDeposition


from utils import resampleDEM 

import matplotlib.pyplot as plt

        

class SourceToSinkSimulator:
    def __init__(self, path_to_topography: str):
        self.path_to_topography = path_to_topography
        self.grid = None
        self.nodata_mask = None
        self.valid_data_mask = None
        self.no_data_value = None
        self.flow_accumulator = None
        self.erosion_deposition_model = None
        self.model_resolution = None

    def createRasterModelGrid(self):
        """Create a RasterModelGrid from a DEM file."""
        with rasterio.open(self.path_to_topography) as src:
            if src.crs.is_geographic:
                # Reproject to UTM if the CRS is geographic
                dst_path = self.path_to_topography.replace('.tif', '_utm.tif')
                self.reprojectToUtm(src, dst_path)
                src = rasterio.open(dst_path)
                print(f"Reprojected raster to UTM and saved as {dst_path}.")
            elevation = src.read(1)
            nodata = src.nodata
            if not nodata:
                if self.no_data_value is not None:
                    nodata = self.no_data_value
                else:
                    print("No nodata value is not found in the raster. Set it before creating the grid.")
                    print("Use SourceToSinkSimulator.setNoData() to set the nodata value.")
            self.nodata_mask = elevation == nodata
            self.valid_data_mask = ~self.nodata_mask
            elevation_filled = fillnodata(elevation, mask=self.nodata_mask, max_search_distance=10)
            self.grid = RasterModelGrid(elevation_filled.shape, xy_spacing=(src.res[0], src.res[1]))
            self.grid.add_field('topographic__elevation', elevation_filled, at='node')
           
            self.model_resolution = src.res[0]  # Assuming square pixels

    def reprojectToUtm(self, src, dst_path):
        """Reprojects a raster to UTM project coordinate system."""

        #find the UTM zone based on the raster's geographic CRS'
        #The determined utm zone corresponds to the center of the raster
        if src.crs.is_geographic:
            lon = (src.bounds.left + src.bounds.right) / 2
            lat = (src.bounds.top + src.bounds.bottom) / 2
            utm_zone = utm.from_latlon(lat, lon)
            utm_crs = f"EPSG:326{utm_zone[2]}" if src.crs.to_epsg() == 4326 else f"EPSG:327{utm_zone[2]}"
        else:
            # If the CRS is already UTM, no reprojection needed
            print("Raster is already in UTM coordinates.")
            return

        # Calculate the transform and metadata for the new UTM projection
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

        # Reproject the raster data
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
        if self.grid is not None:
            self.grid.set_nodata_nodes_to_closed(self.grid.at_node['topographic__elevation'], self.no_data_value)
            try:
                self.grid.set_watershed_boundary_condition(self.grid.field_values('topographic__elevation'))
            except Exception as e:
                print(f"Error setting watershed boundary conditions: {e}")
                print("Seems like there are multiple cells with the lowest elevation.")
                print("Trying to lower the elevation of one of this cells, which will serve as the outlet node.")
                min_elev = self.grid.at_node['topographic__elevation'].reshape(self.grid.shape)[self.valid_data_mask].min()
                self.grid.field_values("topographic__elevation")[self.grid.at_node["topographic__elevation"]==min_elev][0] = min_elev - 1
                self.grid.set_watershed_boundary_condition(self.grid.field_values("topographic__elevation"))
    
    def setPrecipitationRates(self, path_to_precipitation_raster:str):  
        """Set precipitation rates for the simulation."""
        with rasterio.open(path_to_precipitation_raster) as src:
            if src.crs.is_geographic:
                # Reproject to UTM if the CRS is geographic
                dst_path = path_to_precipitation_raster.replace('.tif', '_utm.tif')
                self.reprojectToUtm(src, dst_path)
                src = rasterio.open(dst_path)
                print(f"Reprojected raster to UTM and saved as {dst_path}.")
            if src.res[0] != self.model_resolution:
                print("Warning: The resolution of the precipitation raster does not match the DEM resolution.")
                print("Resampling the precipitation raster to match the DEM resolution.")
                path_to_precipitation_raster = resampleDEM(dst_path, 
                                                            up_scale_factor=src.res[0]/self.model_resolution)
                src = rasterio.open(path_to_precipitation_raster)

            precipitation = src.read(1)
            nodata = src.nodata
            if nodata:
                precipitation[self.nodata_mask] = nodata
            self.grid.add_field('water__unit_flux_in', precipitation, at='node', clobber=True)
            #TODO: need to check the units. for now the units are in mm/year
    
    def setUpErosionDepositionModel(self, 
                                    m_sp:float = 0.45, 
                                    n_sp:float = 1, 
                                    K_sp:float = 0.02, # erodibility coefficient, should later be secified based on geology
                                    v_s:float = 1, #1 means equal contribution from transport and deposition
                                    F_f:float = 0.1, #10% of sediment is permitted to enter the wash load
                                    solver:str = 'adaptive',
                                    runoff_rate: float = None):
        """Set up the erosion and deposition model."""

        if self.grid is None:
            raise ValueError("Grid not created. Call createRasterModelGrid() first.")
        if not self.grid.has_field('water__unit_flux_in') and not runoff_rate:
            raise ValueError("Precipitation rates not set. Call setPrecipitationRates() first or specify runoff_rate keyword argument.")
        
        self.flow_accumulator = FlowAccumulator(self.grid, depression_finder='DepressionFinderAndRouter', flow_director='D8', runoff_rate=runoff_rate)
        self.erosion_deposition_model = ErosionDeposition(self.grid, K_sp=0.01, m_sp=1.0, n_sp=1.0, runoff_rate=runoff_rate)

    def runSimulation(self, years: int = 100, dt: float = 1, uplift_rate: float = 0.002):
        """Run the simulation for a specified number of years."""
        if self.flow_accumulator is None or self.erosion_deposition_model is None:
            raise ValueError("Erosion and deposition model not set up. Call setUpErosionDepositionModel() first.")
        
        n_steps = int(years / dt)
        for i in range(n_steps):
            #accumulate flow a
            self.flow_accumulator.run_one_step()
            #calculate sediment erosion and deposition
            self.erosion_deposition_model.run_one_step(dt=dt)
            #apply uplift rate
            self.grid.at_node['topographic__elevation'][self.valid_data_mask] += uplift_rate * dt

    def visualizeResults(self):
        """Visualize the results of the simulation."""
        if self.grid is None:
            raise ValueError("Grid not created. Call createRasterModelGrid() first.")
        if self.nodata_mask is None:
            raise ValueError("No data mask not created. Call setNoData() first.")
        sed_flux_diff = self.grid.at_node['sediment__influx'] - self.grid.at_node['sediment_outflux']
        sed_flux_diff[self.nodata_mask] = 0
        sed_flux_diff[self.grid.boundary_nodes] = 0
        plt.figure(figsize=(10, 10))
        plt.imshow(sed_flux_diff.reshape(self.grid.shape), cmap='terrain')
        plt.colorbar(label='Sediment Flux Difference (m3/s)')
        plt.show()



    def setNoData(self, nodata_value: float):
        """Set the nodata value for the DEM."""
        self.no_data_value = nodata_value
        print(f"Set no-data value to {self.no_data_value}.")

    def getNoData(self):
        """Get the nodata value."""
        return self.no_data_value
    def getGrid(self):
        """Get the grid."""
        return self.grid
    



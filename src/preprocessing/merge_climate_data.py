import xarray as xr
import os
import numpy as np
import rioxarray
import matplotlib
matplotlib.use('TkAgg')  # Switch to TkAgg backend
import matplotlib.pyplot as plt
from landlab import RasterModelGrid

# 1 Load and Inspect the Dataset
data_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..', 'data', 'GPCP_3.2'))

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Directory does not exist: {data_dir}")

nc_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.nc', '.nc4'))])
if not nc_files:
    raise FileNotFoundError(f"No .nc or .nc4 files found in directory: {data_dir}")

ds = xr.open_mfdataset(nc_files, combine='by_coords', engine='netcdf4', chunks={})
print("Available variables:", list(ds.data_vars))  # Check available variable names

# 2 Subset Spatially (Pamir Plateau Region) - Adjusted range
ds_pamir = ds.sel(lat=slice(42, 32), lon=slice(64, 77))
print(ds_pamir)

# 3 Convert Units to Monthly Totals
precip_daily = ds_pamir['gauge_precip']  # Ensure the correct variable name
time_index = precip_daily['time'].to_index()
days_in_month = np.array(time_index.days_in_month)  # Convert to NumPy array

# Reshape for broadcasting
precip_monthly_tot = precip_daily * days_in_month[:, None, None]
precip_monthly_tot.name = 'precip_monthly_mm'

# 4 Create a 33-Year Average or Climatology
precip_annual = precip_monthly_tot.groupby('time.year').sum('time')
precip_annual.name = 'precip_annual_mm'
precip_annual_33yr = precip_annual.sel(year=slice(1990, 2023))
precip_annual_avg_33yr = precip_annual_33yr.mean(dim='year')

ds_30yr = precip_monthly_tot.sel(time=slice('1990-01', '2023-12'))
precip_climatology_monthly = ds_30yr.groupby('time.month').mean(dim='time')
precip_climatology_monthly.name = 'precip_climo_mm'

# Assign coordinate system and set spatial dims for climatology
precip_climatology_monthly.rio.write_crs("EPSG:4326", inplace=True)
precip_climatology_monthly = precip_climatology_monthly.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

# Reproject to UTM Zone 42N
precip_utm = precip_climatology_monthly.rio.reproject("EPSG:32642")

mm_per_year_to_mm_per_hour = 1.0 / (365 * 24)
precip_hourly = precip_annual_avg_33yr * mm_per_year_to_mm_per_hour

# Define grid (assuming ny, nx from data shape)
ny, nx = precip_annual_avg_33yr.shape
grid_spacing = 1000  # Example: 1 km
mg = RasterModelGrid((ny, nx), xy_spacing=grid_spacing)
rainfall_rate = precip_annual_avg_33yr.values.flatten()

# Add field with updated syntax
mg.add_field("rainfall__rate", rainfall_rate, at="node", units="mm/yr")

# Save to TIFF
output_dir = "C:/Users/loiq.amonbekov/PycharmProjects/s2s-modeling/data/raw"
output_file = os.path.join(output_dir, "precip_annual_avg_33yr.tif")

# Ensure the output directory exists (create it if it doesn’t)
os.makedirs(output_dir, exist_ok=True)

# Assign CRS, set spatial dims, and save to the specific folder
precip_annual_avg_33yr = precip_annual_avg_33yr.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
precip_annual_avg_33yr.rio.write_crs("EPSG:4326", inplace=True)
precip_annual_avg_33yr.rio.to_raster(output_file)

# Plot
precip_annual_avg_33yr.plot()
plt.title("Mean Annual Precipitation (1991–2020)")
plt.show()
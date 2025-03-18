# src/main.py

from src.preprocessing.load_dem import load_dem, fill_nodata
from src.visualization.plot_dem import visualize_dem
from src.config import dem_file
import os

def main():
    print(f"🔹 Checking DEM file at: {dem_file}")
    if not os.path.exists(dem_file):
        print(f"❌ ERROR: File not found at {dem_file}. Check the path!")
        return

    print("✅ DEM file found. Loading data...")
    dem_data, dem_meta, nodata_mask = load_dem(dem_file)

    # Check for NoData and fill if needed
    if nodata_mask is not None and nodata_mask.any():
        print("⚠️ NoData values detected. Filling missing values...")
        dem_data = fill_nodata(dem_data, nodata_mask)
    else:
        print("✅ No NoData values found in DEM.")

    # Visualize the DEM in 3D
    visualize_dem(dem_data, dem_meta)

if __name__ == "__main__":
    main()

import os
import numpy as np
import rasterio

# Define base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

def load_precipitation(precip_path):
    """Load precipitation data from a .tif file."""
    with rasterio.open(precip_path) as src:
        precip = src.read(1)  # First band
        bounds = src.bounds
        res = src.res
    return precip, bounds, res

def preprocess_precipitation(precip):
    """Basic preprocessing: mask negative values (if any) and normalize."""
    precip = np.where(precip < 0, 0, precip)  # Remove invalid data
    return precip

if __name__ == "__main__":
    precip_path = os.path.join(DATA_DIR, 'precipitation.tif')  # Adjust filename as needed
    precip, bounds, (dx, dy) = load_precipitation(precip_path)
    print(f"Precipitation loaded. Shape: {precip.shape}, Resolution: ({dx}, {dy})")

    precip_clean = preprocess_precipitation(precip)
    print(f"Precipitation preprocessed. Min: {precip_clean.min()}, Max: {precip_clean.max()}")
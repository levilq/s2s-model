# src/config.py

import os

# Get absolute path to the project directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Data Paths
data_dir = os.path.join(BASE_DIR, "data")
raw_data_dir = os.path.join(data_dir, "raw")
processed_data_dir = os.path.join(data_dir, "processed")
results_dir = os.path.join(BASE_DIR, "results")

# DEM and Climate Data Paths
dem_file = os.path.join(raw_data_dir, "SarezWaterShed.tif")
climate_file = os.path.join(raw_data_dir, "climate.csv")

# Model Parameters
flow_director = "D8"  # Flow direction method
time_step = 1.0  # Time step for Landlab models

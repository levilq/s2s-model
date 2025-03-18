# src/visualization/plot_dem.py

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Set backend to ensure proper rendering in PyCharm
matplotlib.use('TkAgg')  # 'TkAgg' works well for interactive plots


def visualize_dem(dem_data, meta):
    """
    Visualize the DEM as a 2D heatmap.
    :param dem_data: DEM array
    :param meta: DEM metadata (for spatial extent)
    """
    # Create coordinate grid based on DEM dimensions
    nrows, ncols = dem_data.shape
    x = np.linspace(meta['transform'][2],
                    meta['transform'][2] + meta['width'] * meta['transform'][0],
                    ncols)
    y = np.linspace(meta['transform'][5],
                    meta['transform'][5] + meta['height'] * meta['transform'][4],
                    nrows)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Display DEM as a heatmap
    extent = [x.min(), x.max(), y.min(), y.max()]
    img = ax.imshow(dem_data, cmap='terrain', extent=extent, origin="upper")

    # Labels and title
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("2D Digital Elevation Model (DEM)")

    # Add colorbar
    plt.colorbar(img, label="Elevation (m)")

    # Show the plot interactively
    plt.show(block=True)  # Ensures the plot remains open

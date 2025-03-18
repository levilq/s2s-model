import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, Resampling

# File paths
input_file = "C:/Users/loiq.amonbekov/PycharmProjects/s2s-modeling/data/raw/precip_annual_avg_33yr.tif"
output_file = "C:/Users/loiq.amonbekov/PycharmProjects/s2s-modeling/data/raw/precip_annual_avg_33yr_resampled.tif"

# Desired pixel size in meters
target_pixel_size = 30.0

# Open the input raster
with rasterio.open(input_file) as src:
    original_crs = src.crs  # Get the original CRS
    print(f"Original CRS: {original_crs}")

    # Ensure the raster is in a projected coordinate system (meters)
    if src.crs.is_geographic:
        print("📌 Detected Geographic CRS. Reprojecting to a projected CRS for proper scaling.")

        # Choose a projected CRS (e.g., UTM)
        dst_crs = "EPSG:32643"  # Example: UTM Zone 43N, change based on your location

        # Compute the transformation for the new CRS
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=target_pixel_size
        )

        # Create a new profile with the projected CRS
        profile = src.profile.copy()
        profile.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        # Reproject the raster
        with rasterio.open(output_file, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear
                )

        print(f"✅ Reprojected raster saved as: {output_file}")

    else:
        print("📌 Raster is already in a projected CRS. Proceeding with resampling.")

    # Open the reprojected raster to apply resampling
    with rasterio.open(output_file) as src_projected:
        new_width = int(src_projected.width * (src_projected.res[0] / target_pixel_size))
        new_height = int(src_projected.height * (src_projected.res[1] / target_pixel_size))

        print(f"Resampling to: {new_width} x {new_height}")

        # Compute the new transformation matrix
        new_transform = src_projected.transform * src_projected.transform.scale(
            (src_projected.width / new_width),
            (src_projected.height / new_height)
        )

        # Update profile for resampling
        profile = src_projected.profile
        profile.update({
            "height": new_height,
            "width": new_width,
            "transform": new_transform,
            "dtype": src_projected.dtypes[0]
        })

        # Read and resample the entire raster
        data = src_projected.read(
            out_shape=(src_projected.count, new_height, new_width),
            resampling=Resampling.bilinear
        )

        # Save the resampled raster
        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(data)

print(f"✅ Resampled raster saved as: {output_file}")

# Verify new pixel size
with rasterio.open(output_file) as resampled:
    print(f"New Pixel Size: {resampled.res}")

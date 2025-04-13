
import numpy as np
import rasterio
from scipy.ndimage import zoom
from rasterio.enums import Resampling


def resampleDEM(dem, up_scale_factor=None, target_resolution=None):
    """
    Resample the DEM to a target resolution using bilinear interpolation.
    
    Parameters:
    dem (str): Path to the input DEM.
    up_scale_factor (float): Scale factor for upscaling the DEM. For downsampling, use a value < 1 (e.g. 1/2).
    If None, target_resolution must be provided. 
    target_resolution (float): The target resolution for resampling.
    
    Returns:
    str: Path to the resampled DEM.
    """

    if up_scale_factor is None and target_resolution is None:
        raise ValueError("Either up_scale_factor or target_resolution must be provided.")
    if target_resolution is not None:
        with rasterio.open(dem) as src:
            original_resolution = src.res[0]
            up_scale_factor = target_resolution / original_resolution
            data = src.read(
                1, out_shape=(int(src.height * up_scale_factor), int(src.width * up_scale_factor)),
            resampling = Resampling.bilinear
            )

            transform = src.transform * src.transform.scale(
                (src.width / data.shape[-1]), (src.height / data.shape[-2])
            )
            crs = src.crs
    else:
        with rasterio.open(dem) as src:
            assert up_scale_factor is not None, "Either up_scale_factor or target resolution must be specified."
            original_resolution = src.res[0]
            target_resolution = original_resolution * up_scale_factor
            data = src.read(
                1, out_shape=(int(src.height * up_scale_factor), int(src.width * up_scale_factor)),
            resampling = Resampling.bilinear
            )

            transform = src.transform * src.transform.scale(
                (src.width / data.shape[-1]), (src.height / data.shape[-2])
            )
            crs = src.crs

    # Write the resampled DEM to a new file
    output_path = dem.replace('.tif', f'_resampled_{up_scale_factor}.tif')
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)
        print(f"Resampled DEM saved at: {output_path} from {original_resolution} to {target_resolution} m")


    return output_path


if __name__ == "__main__":
    # Example usage
    dem_path = r'path//to/your//dem.tif'  # Replace with your DEM path
    up_scale_factor = 2.0  # Example scale factor for upscaling
    target_resolution = None  # Set to None if using up_scale_factor

    resampled_dem_path = resampleDEM(dem_path, up_scale_factor, target_resolution)
    print(f"Resampled DEM saved at: {resampled_dem_path}")
"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer.

Date Created: May 2024

Date Modified: August 2024

About: TIFF functions that do the following:
        - Read GeoTIF files.
        - Acquire spatial information of the TIF (CRS, bounds, affine transformation).
            --> Requires read GeoTIF function prior.
        - Convert pixels to geographic coordinates.
        - Convert coordinates containing pixel values to GeoDataFrames via GeoParquets (parallel processed).
            --> Requires convert pixels to geographic coordinates function prior.
"""
import rasterio
from rasterio.crs import CRS
import numpy as np
from typing import List
from tqdm import tqdm
from pandas import DataFrame
from geopandas import GeoDataFrame
from shapely.geometry import Point
from multiprocessing import cpu_count
from functools import partial
from .essentials import ParallelPool
from .data_utils import DataEng


class GeoTIFF_Tools:

    def __init__(self):
        pass

    def read_tif(self, image_file: str) -> rasterio.DatasetReader:
        """
        Read the TIF file.

        :param image_file: The name of the file

        :return: Rasterio DatasetReader.
        """

        tif_file = rasterio.open(image_file)
        return tif_file

    def acq_spatialinf(self, image: rasterio.DatasetReader) -> List:
        """
        Acquire spatial information of the GeoTIFF image.
        These are: the CRS, bounds, and the affine transformation.

        Pre-requisites: Must use the "read_tif(image_file=)" function or rasterio.open().

        :param image: Rasterio image that is read as DatasetReader.

        :return: List - CRS, bounds, and affine transformation.
        """
        try:
            crs  = image.crs.data['init']
        except KeyError:
            crs  = CRS.from_wkt(image.crs.wkt).to_proj4()

        bnds     = image.bounds
        transfrm = image.transform

        return [crs, bnds, transfrm]

    def pixel2coord(self, image: rasterio.DatasetReader) -> DataFrame:
        """
        Convert the pixel coordinates to geographic coordinates.

        :param image: Rasterio image that is read as DatasetReader.

        :return: DataFrame with x and y in a list per row. This will require
                 the explode function in Pandas to expand and likely parallel processing
                 for any downstream processes.
        """

        height, width = image.shape
        cols, rows    = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys        = rasterio.transform.xy(image.transform, rows, cols)

        df = DataFrame({"x" : np.array(xs), "y" : np.array(ys), "x_pixel" : list(rows), "y_pixel" : list(cols)})

        return df

    def coord2gdf(self, df: DataFrame, crs: str, output: str):
        """
        Coordinates to GeoDataFrame parallel processed - outputs are in smaller partitions to prevent out-of-memory (OOM)
        issues.

        :param df: DataFrame.
        :param crs: Coordinate Reference System in EPSG. E.g., "epsg:4326".
        :param output: Output directory with main filename without extension.
        """

        DataEng.checkdir(dir_name=output)
        split_dfs    = np.array_split(df, cpu_count())
        i            = 0 
        for s in tqdm(split_dfs):            
            parallel_set = np.array_split(s, cpu_count())
            parallel_set = [[p, f"{i}_{c}"] for p,c in zip(parallel_set, range(cpu_count()))]
            partial_func = partial(self._parallel_coord2gdf, crs = crs, output = output)
            ParallelPool(start_method='spawn', partial_func = partial_func, main_list = parallel_set)
            i += 1

    def _parallel_coord2gdf(self, parallel_set, crs, output):
        """
        Function during parallel processing that converts coordinates to GeoDataFrame.

        :param parallel_set: Containing DataFrame and filename structure.
        :param crs: Coordinate Reference System in EPSG. E.g., "epsg:4326".
        :param output: Output directory with main filename without extension.

        :return: Indirectly saving outputs as GeoParquet files.
        """

        tqdm.pandas()

        part_df  = parallel_set[0].explode(["x", "y", "x_pixel", "y_pixel"])
        part_num = parallel_set[1]

        part_df['pnt'] = part_df[['x', 'y']].progress_apply(lambda e: Point(*e), axis=1)
        part_gdf = GeoDataFrame(data = part_df, crs = crs, geometry = 'pnt')
        part_gdf.to_parquet(path = f"{output}/{DataEng.filename_fromdir(output)}_{part_num}.geoparquet", index=False, compression='zstd')
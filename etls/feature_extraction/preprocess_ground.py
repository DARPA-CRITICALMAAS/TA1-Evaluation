"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer

Date Created: October 2024

Last Update: October 2024

About: Pre-processing ground-truth of feature extraction to optimize computation (i.e., parallel processing) when
       evaluating vector (i.e., IoU and spatial index matching). Optimize computation during evaluation is intended to
       reduce compute time and vastly mitigate the number of combinations cross-comparing between per ground-truth
       feature to per predicted / inferenced feature from the performers. In other words, preventing to have a naive
       1-to-1 comparison. This would be truncated by partitioning out the ground-truth data per ID into geologic values
       (i.e., labels) and have them spatially-indexed to reduce computation searches. Additionally, with this approach
       is to have an apples-to-apples comparison between the ground-truth and inferenced feature; thus, a much more
       precise way to evaluate.

Pre-Requisites: Must run PrepEval class beforehand (if have not saved the data.match_binary_vector variable as a feather
                file); otherwise, use the feather file containing the following schema:
                    - id                => str or int => unique identifier that ties to the specific geologic map.
                    - binary_value      => list       => List containing geologic values derived from the naming of the
                                                         binary rasters.
                    - geo_value_vector  => list       => List containing geologic values matched from each binary_value
                                                         and is from the shapefile (ground truth vector data).
                    - field_name        => str or int => Name of the field where the binary_value and geo_value_vector
                                                         had a high threshold match.
                    - matched_ratio     => list       => List of threshold values (closer to 1 indicates 100% match)
                                                         from each binary_value in the list to the corresponding
                                                         geo_value_vector in the list.
                    - shp_file          => str        => Name of the shapefile (ground-truth) where the data has been
                                                         matched and will be used to read and process.

Outputs: FOR FEATURE EXTRACTION EVALUATION VECTOR METRICS
    - ground_truth_schema (parquet file) --- main file to be used in preparation for feature extraction vector eval.
        - id                            => str or int   => unique identifier that ties to the specific geologic map.
        - ground_truth                  => str          => Name of the shapefile where the data has been matched and
                                                           been used to read and process.
        - field_name                    => str or int   => Name of the field where the binary_value and geo_value_vector
                                                           had a high threshold match and was used during partition process.
        - use_field_in_part             => str          => Field in the partitioned file to be used that contains geo_value_vector.
        - geom_field_in_part            => str          => Geometry field in the partitioned file to be used.
        - geom_size_file                => str          => Path to the ground truth schema parquet file.

    - *ground_truth_geom_size (parquet files) -- parquet file per ID per field_name that was then partitioning out the
                                                 corresponding geo_value_vector values. Contains crucial information to
                                                 orchestrate optimal parallel processing for the feature extraction vector eval.
        - id                => str or int   => unique identifier that ties to the specific geologic map.
        - binary_value      => str (likely) => Containing geologic value derived from the naming of the binary rasters.
        - geo_value_vector  => str (likely) => Matched geologic value from the binary_value derived from the shapefile.
        - matched_ratio     => float or int => Threshold value (closer to 1 indicates 100% similarity match) between the
                                               binary_value and geo_value_vector.
        - count             => int          => The number of ground truth features per geo_value_vector value. Crucial to
                                               optimize parallel processing.
        - partition_file    => str          => Name of the partitioned geoparquet file to be used during the feature eval.
                                               vector metrics.

    - *sub_geog (geoparquet partitioned files) -- partitioned geoparquet files for each geo_value_vector value (i.e.,
                                                  containing the ground_truth).
        - geo_value_vector  => str (likely)     => Geologic value from the shapefile that was matched to the binary_value.
        - min_h3            => list             => List of H3 spatial indices at a defined minimum (coarser) resolution.
        - max_h3            => list             => List of H3 spatial indices at a defined maximum (finer) resolution.
        - min_res           => int              => Minimum (coarser) resolution number.
        - max_res           => int              => Maximum (finer) resolution number.
        - geom_type         => str              => Geometry type (e.g., point, line, polygon).
        - geom              => Shapely geometry => Geometry of the feature.

    - spatial_index (feather files) -- spatial index table per geo-partitioned (i.e., geologic value).
"""

# Data-Engineering Related Packages
from geopandas import read_file, GeoDataFrame
from pandas import DataFrame, read_parquet, concat
from numpy import array_split
import re

# Miscellaneous packages
from tqdm import tqdm
from typing import List, Union
from multiprocessing import cpu_count, Manager
from functools import partial

# Packages used to import custom-made packages outside of relative path
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Custom-made packages
from ..utils import SpatialOps, ParallelPool, DataEng


class PreprocessVector:

    def __init__(self, match_binary: Union[str, DataFrame], crude_match_binary: Union[str, DataFrame], output_dir, min_res: int = 6, max_res: int = 9):
        """
        :param match_binary: DataFrame or string (path to parquet file) containing ID, binary_value, geo_value_vector,
                                matched_ratio, matching field, and shapefile. This information must be acquired from the
                                PrepEval class process.
        :param crude_match_binary: DataFrame or string (path to parquet file) containing ID, binary_value, and shapefile.
                                   Crude match indicates potential list of matches.
        :param output_dir: Main path to export schema and partitioned geoparquet files. Typically, this must be the
                           main feature path folder.
        :param min_res: Default to 6; The minimum spatial index resolution for H3.
        :param max_res: Default to 9; The maximum spatial index resolution for H3.
        """

        # Important DataFrames
        self.match_binary       = self._read_df(df=match_binary).reset_index().drop(columns=['index'])
        self.crude_match_binary = self._read_df(df=crude_match_binary).reset_index().drop(columns=['level_0', 'level_1'])
        self.crude_grp = (
            self.crude_match_binary
            .groupby(['id', 'geom_type', 'shp_file'], as_index=False)
            .agg({'unmatched_binary' : lambda x: x.tolist()})
            .reset_index()
        )

        self.crude_grp.to_parquet(f"{output_dir}/ft_crude_grp.parquet")

        # Make the minimum and maximum resolution and main output directory universal
        self.min_res    = min_res
        self.max_res    = max_res
        self.output_dir = output_dir

        if len(self.match_binary) > 0:
            self.match_binary_grnd_schema = self._parallel_execution(df=self.match_binary, match_partition='matched')
            self.match_binary_grnd_schema.to_parquet(f"{output_dir}/ft_match_binary_grnd_schema.parquet")

        if len(self.crude_grp) > 0:
            self.crude_match_grnd_schema = self._parallel_execution(df=self.crude_grp, match_partition='crude_match')
            self.crude_match_grnd_schema.to_parquet(f"{output_dir}/ft_crude_match_grnd_schema.parquet")

    def _parallel_execution(self, df: DataFrame, match_partition: str = Union["matched", "crude_match"]):
        # Split the DataFrame containing either matched binary geologic vector values or crude-based into N number of CPUs.
        # Instantiate parallel processing

        # Split the DataFrame containing the matched binary geologic vector values into N number of CPUs.
        # Instantiate parallel processing
        split_dfs       = array_split(df, cpu_count())
        L1              = Manager().list()
        partial_func    = partial(self._partition_geologic_value, L1=L1, match_partition=match_partition)
        ParallelPool(start_method='spawn', partial_func=partial_func, main_list=split_dfs)

        # Concatenate ground-truth schema information into one DataFrame and export as a parquet file.
        L1 = concat(L1)
        grnd_truth_schema = f"{self.output_dir}/{match_partition}_ground_truth_schema.parquet"
        L1.to_parquet(grnd_truth_schema)

        return L1

    def _read_df(self, df: Union[str, DataFrame]):
        # If it is a string, assumes you are importing a CSV file that needs to be read.
        if isinstance(df, str):
            df = read_parquet(df)

        # Deep copy (i.e., overwrite), reset index (would impact the iloc function) and drop a column
        #df = df.copy(deep=True).reset_index().drop(columns=['index'])
        return df

    def _partition_geologic_value(self, split_df: DataFrame, L1, match_partition):
        """

        :param split_df:
        :param L1:
        :param match_partition:

        :return:
        """

        self.f_cols_match = ['binary_value', 'geo_value_vector', 'matched_ratio']
        self.id_col       = ['id']
        self.f_cols_crude = ['unmatched_binary']

        """
        Lambda functions:
            - spatial_index    => auto acquire H3 spatial index based on the geometry - would be used in map function. 
            - break_multi_geom => break multi-geometries where applicable. 
            - regex_clean      => regex clean 
            - convert_str_dict => converting str to dictionary
        """

        self.spatial_index    = lambda x: SpatialOps().auto_acqH3_fromgeom(geom=x, min_res=self.min_res, max_res=self.max_res)
        self.break_multi_geom = lambda x: SpatialOps().break_multi_geom(geom=x)
        self.regex_clean      = lambda x: re.sub(r'\W+', '', x)
        self.convert_str_dict = lambda x: str(x)

        # Iterate through each row of the partitioned DataFrame
        col_names = split_df.columns
        for s in tqdm(split_df.itertuples(index=False), total=split_df.shape[0]):

            self.shp_file = s.shp_file  # name and path to the shapefile to read
            self.id_val   = s.id  # ID value used to track
            tmp_data      = self._specific_processes(row_data=s, match_partition=match_partition, col_names=col_names)

            if tmp_data is not None:
                L1.append(tmp_data)

    def _specific_processes(self, row_data, match_partition, col_names):

        tqdm.pandas()

        # Read shapefile, obtain geometry field, and its original CRS
        orig_gdf   = read_file(filename = self.shp_file)
        geom_field = orig_gdf.geometry.name
        orig_crs   = orig_gdf.crs
        refine_gdf = None

        if match_partition == "matched":
            keep_cols  = ['id', 'binary_value', 'geo_value_vector', 'matched_ratio', 'count']
            field_name = row_data.field_name

            # Series to DataFrame and transposing to then explode (i.e., expand) the binary value,
            # geo_value_vector, and matched ratio since they are a list. This will be used to
            # merge with the matched shapefile.
            tmp_df = (
                DataFrame(row_data)
                .transpose()
                .pipe(lambda a: a.rename(columns={c:r for c,r in zip(a.columns, col_names)}))
                .explode(self.f_cols_match)
                [self.id_col + self.f_cols_match]
            )

            try:
                refine_gdf = (
                    orig_gdf
                    # Retain only the field of interest and geometry field
                    [[field_name, geom_field]]
                    # Convert to EPSG 4326 to identify H3 index
                    .to_crs('epsg:4326')
                    # Grouping by the geologic values in the field of interest - first, we need to break
                    # any multi-geometries or geometries containing Z (i.e., 3D) and auto-expand if necessary.
                    .groupby([field_name], as_index=False)
                    .progress_apply(lambda a: a.assign(geom = lambda d: list(map(self.break_multi_geom, d[geom_field]))))
                    .explode('geom')
                    # Grouping by the geologic values in the field of interest, acquire the amount of ground-truth per
                    # geologic values and auto acquiring spatial index information via the map lambda function.
                    .groupby([field_name], as_index=False)
                    .progress_apply(lambda a: a.assign(count = len(a),
                                                       inf   = lambda b: list(map(self.spatial_index, b['geom']))))
                    .reset_index()
                    # Concat with the original GeoDataFrame with the spatial index information.
                    # (H3 @ min res., H3 @ max res., min res., max res., number, geometry type)
                    .pipe(lambda a: concat([a, DataFrame(list(a['inf']), columns=['min_h3', 'max_h3', 'min_res',
                                                                                  'max_res', 'geom_type'])], axis=1))
                    # Merge to incorporate binary value and matched ratio with the geo_value_vector via the field of interest
                    .merge(tmp_df, left_on=[field_name], right_on=['geo_value_vector'])
                )

                # Delete orig_gdf to reduce memory footprint while parallel-processing
                del orig_gdf

            except (ValueError, TypeError) as e:
                print(e, self.id_val, field_name, self.shp_file)
                return None

        elif match_partition == "crude_match":
            keep_cols  = None
            field_name = None

            # Series to DataFrame and transposing to then explode (i.e., expand) the unmatched_binary value,
            # since they are a list. This will be used to merge with the matched shapefile.
            tmp_df = (
                DataFrame(row_data)
                .transpose()
                .pipe(lambda a: a.rename(columns={c: r for c, r in zip(a.columns, col_names)}))
                .explode(self.f_cols_crude)
                [self.id_col + self.f_cols_crude + ['index']]
            )

            try:
                refine_gdf = (
                    orig_gdf
                    # Create a fake field
                    .assign(pseudo_idx = 1,
                            reference  = lambda a: a['pseudo_idx'].cumsum()-1)
                    # Retain only the field of interest and geometry field
                    [['reference', geom_field]]
                    # Convert to EPSG 4326 to identify H3 index
                    .to_crs('epsg:4326')
                    # Grouping by the geologic values in the field of interest - first, we need to break
                    # any multi-geometries or geometries containing Z (i.e., 3D) and auto-expand if necessary.
                    .groupby(['reference'], as_index=False)
                    .progress_apply(lambda a: a.assign(geom=lambda d: list(map(self.break_multi_geom, d[geom_field]))))
                    .explode('geom')
                    # Grouping by the geologic values in the field of interest, acquire the amount of ground-truth per
                    # geologic values and auto acquiring spatial index information via the map lambda function.
                    .groupby(['reference'], as_index=False)
                    .progress_apply(lambda a: a.assign(count=len(a),
                                                       inf=lambda b: list(map(self.spatial_index, b['geom']))))
                    .reset_index()
                    # Concat with the original GeoDataFrame with the spatial index information.
                    # (H3 @ min res., H3 @ max res., min res., max res., number, geometry type)
                    .pipe(lambda a: concat([a, DataFrame(list(a['inf']), columns=['min_h3', 'max_h3', 'min_res',
                                                                                  'max_res', 'geom_type'])], axis=1))
                )

                # Delete orig_gdf to reduce memory footprint while parallel-processing
                del orig_gdf

            except (ValueError, TypeError) as e:
                print(e, self.id_val, self.shp_file)
                return None

        if refine_gdf is not None:
            shp_file_name   = DataEng.filename_fromdir(dir_file=self.shp_file)
            partition_dir   = f"{self.output_dir}/{self.id_val}/partition/{match_partition}/{shp_file_name}_{len(tmp_df)}"
            DataEng.checkdir(dir_name=partition_dir)
            geom_size_label = f"{partition_dir}/ground_truth_geom_size.parquet"

            if keep_cols is not None:
                use_field_in_part = 'geo_value_vector'

                geom_size_df = (
                    refine_gdf
                    [keep_cols]
                    .drop_duplicates(keep_cols)
                    # change binary value to geo_value in case
                    .assign(match_binary     = True,
                            geo_value_vector = lambda a: list(map(self.regex_clean, a[use_field_in_part])),
                            partition_file   = lambda a: a[[use_field_in_part]].apply(lambda e: ''.join(map(str, [partition_dir, '/', *e, "/sub_geog.geoparquet"])), axis=1))
                )

                geom_size_df.to_parquet(geom_size_label)

                (
                    refine_gdf
                    [['binary_value', use_field_in_part, 'min_h3', 'max_h3', 'min_res', 'max_res', 'geom_type', 'geom']]
                    .assign(geo_value_vector = lambda d: list(map(self.regex_clean, d[use_field_in_part])))
                    .pipe(lambda a: SpatialOps().pipe2gdf(df = a, geom_field='geom', epsg='epsg:4326'))
                    .groupby([use_field_in_part], as_index=False) # Change here to geo_value_vector in case
                    .apply(lambda a: self._export_partition(tmp_gdf          = a,
                                                            orig_crs         = orig_crs,
                                                            tmp_df           = None,
                                                            partition_export = partition_dir,
                                                            match_binary     = True))
                )

            else:
                use_field_in_part = None

                geom_size_df = (
                    tmp_df.assign(match_binary     = False,
                                  geo_value_vector = None,
                                  partition_file   = lambda a: a[['unmatched_binary']].apply(lambda e: ''.join(map(str, [partition_dir, "/sub_geog.geoparquet"])), axis=1),
                                  count            = refine_gdf['count'].sum())
                )

                (
                    refine_gdf
                    [['min_h3', 'max_h3', 'min_res', 'max_res', 'geom_type', 'geom']]
                    .pipe(lambda a: SpatialOps().pipe2gdf(df = a, geom_field='geom', epsg='epsg:4326'))
                    .pipe(lambda a: self._export_partition(tmp_gdf          = a,
                                                           orig_crs         = orig_crs,
                                                           tmp_df           = tmp_df,
                                                           partition_export = partition_dir,
                                                           match_binary     = False))
                )

                geom_size_df.to_parquet(geom_size_label)

            # Append ground-truth schema - i.e., per ID, etc. This will be used to help orchestrate for the eval. metrics.
            tmp_schema = DataFrame({
                "id"                 : [self.id_val],
                "ground_truth"       : [self.shp_file],
                "field_name"         : [field_name],
                "use_field_in_part"  : [use_field_in_part],
                "geom_field_in_part" : ["geom"],
                "geom_size_file"     : [geom_size_label],
                "match_binary"       : [match_partition]
            })

            return tmp_schema

        else:
            return None

    def _export_partition(self, tmp_gdf: GeoDataFrame, tmp_df: Union[DataFrame, None], orig_crs, partition_export, match_binary: bool):
        """

        :param tmp_gdf:
        :param orig_crs:
        :param partition_export:
        :param match_binary:

        :return:
        """

        if match_binary:
            unique_value = tmp_gdf['geo_value_vector'].iloc[0] # originally geo_value_vector - change here in case
            output_path  = f"{partition_export}/{unique_value}"
            DataEng.checkdir(dir_name=output_path)
            SpatialOps().geoparquet(gdf         = tmp_gdf.to_crs(orig_crs),
                                    output_path = f"{output_path}/sub_geog.geoparquet")

        else:
            #unique_value = tmp_df['index'].iloc[0]
            output_path  = f"{partition_export}"
            DataEng.checkdir(dir_name=output_path)
            SpatialOps().geoparquet(gdf         = tmp_gdf.to_crs(orig_crs),
                                    output_path = f"{output_path}/sub_geog.geoparquet")

        #try:
        #    spatial_index = (
        #        SpatialOps()
        #        .build_h3_index_table(tmp_gdf      = tmp_gdf,
        #                              unique_value = unique_value,
        #                              output_path  = output_path)
        #        .assign(table = lambda d: list(map(self.convert_str_dict, d['table'])))
        #        .reset_index()
        #        .drop(columns=['level_0'])
        #    )

        #    spatial_index.to_feather(f"{output_path}/spatial_index.feather")

        #except IndexError:
        #    print(f"Failed to index: {output_path}")
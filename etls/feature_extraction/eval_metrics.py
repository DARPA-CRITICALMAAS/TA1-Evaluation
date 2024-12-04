"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer


"""

import numpy as np
from pandas import read_parquet, DataFrame, concat, read_csv, Series
from geopandas import read_parquet as gpq, GeoDataFrame
from shapely import overlaps
from shapely.geometry import Polygon, LineString, Point
from typing import Union, List
from tqdm import tqdm

from multiprocessing import Manager, cpu_count
from functools import partial
import warnings

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Custom packages to import
from ..utils import SpatialOps, EquitDfs, ParallelPool, DataEng


class FeatureEval:

    def __init__(self,
                 grnd_match_schema: Union[DataFrame, str],
                 grnd_crude_schema: Union[DataFrame, str, None],
                 inf_polygon_schema: Union[DataFrame, str, None],
                 inf_line_schema: Union[DataFrame, str, None],
                 inf_point_schema: Union[DataFrame, str, None],
                 feat_tif_file: Union[DataFrame, str],
                 eval_set: Union[DataFrame, str],
                 output_dir: str,
                 evaluate_feature: Union['point', 'line', 'polygon', 'all'],
                 eval_id_field: str     = "ID",
                 eval_cog_id_field: str = "COG ID",
                 eval_scale_field: str  = "Scale",
                 dynamic_buffer: float  = 0.01):

        self.output_dir        = output_dir
        self.eval_id_field     = eval_id_field
        self.eval_cog_id_field = eval_cog_id_field
        self.eval_scale_field  = eval_scale_field
        self.dynamic_buffer    = dynamic_buffer

        # Ground-truth Data Information
        self.grnd_match_schema  = self._read_data(df=grnd_match_schema, to_concat=True)
        self.grnd_ids           = self.grnd_match_schema['id'].unique()
        self.grnd_crude_schema  = self._read_data(df=grnd_crude_schema)
        self.feat_tif_file      = (
            self._read_data(df=feat_tif_file)
            .query('has_file == True')
            .drop_duplicates(['id'])
            [['id', 'path']]
        )

        self.eval_set = self._read_data(df=eval_set)

        # Performer Data Information
        self.inf_polygon_schema = self._read_data(df=inf_polygon_schema)
        self.inf_line_schema    = self._read_data(df=inf_line_schema)
        self.inf_point_schema   = self._read_data(df=inf_point_schema)
        self.unique_cols_grp    = ['id', 'cog_id', 'system', 'system_version',
                                   'feature_type', 'message', 'data',
                                   'georeferenced_data', 'legend_data_added']

        self.direct_query = 'valid_synops == True and missing_val.notna() and num_missing.notna() and perc_missing < 100 ' \
                            'and orch_info.notna() and size_df > 0'

        indirect_query = 'valid_synops == True and missing_val.notna() and num_missing.notna() and perc_missing == 100'

        if evaluate_feature == "polygon":
            self.polygon_eval_data = self._polygon_eval()

        elif evaluate_feature == "line":
            self.line_eval_data = self._line_eval()

        elif evaluate_feature == "point":
            self.point_eval_data = self._point_eval()

        elif evaluate_feature == "all":
            self.polygon_eval_data = self._polygon_eval()
            self.line_eval_data    = self._line_eval()
            self.point_eval_data   = self._point_eval()

    # these functions (polygon_eval, point_eval, line_eval) can be into one; however, there is an issue with the
    # unmatched / uncertain polygons that need to be addressed.
    def _polygon_eval(self) -> Union[List, None]:
        if self.inf_polygon_schema is not None:
            inf_polygon_orch         = self._build_orchestration_match(inf_schema=self.inf_polygon_schema)
            direct_eval_polygon      = inf_polygon_orch.query(self.direct_query)
            failed_eval_polygon      = inf_polygon_orch.loc[inf_polygon_orch.index.difference(direct_eval_polygon.index.tolist())]

            # Direct match parallelization metrics - 23 hrs. 55 min. 34 sec.
            if len(direct_eval_polygon) > 0:
                direct_orch_info_polygon = self._concat_orch_data(df=direct_eval_polygon)
                direct_polygon_metrics   = self._execute_parallel(df_info      = direct_orch_info_polygon,
                                                                  direct_eval  = direct_eval_polygon,
                                                                  use_eval_set = False)

                return [inf_polygon_orch, direct_eval_polygon, failed_eval_polygon, direct_orch_info_polygon, direct_polygon_metrics]

            # Unmatched / crude-match parallelization metrics - does not work (leave commented until investigated)
            #if len(self.failed_eval_polygon) > 0:
            #    self.failed_eval_polygon_metrics = self.failed_eval_polygon.query(indirect_query)
            #    self.crude_orch_info_polygon     = self._build_orchestration_crude(failed_eval = self.failed_eval_polygon_metrics,
            #                                                                       feat_type   = "polygon")

        else:
            return None

    def _point_eval(self) -> Union[List, None]:

        # Points Performer Data - if applicable - 6 hrs. 5 min. 58 sec.
        if self.inf_point_schema is not None:
            inf_point_orch         = self._build_orchestration_match(inf_schema=self.inf_point_schema)
            direct_eval_point      = inf_point_orch.query(self.direct_query)
            failed_eval_point      = inf_point_orch.loc[inf_point_orch.index.difference(direct_eval_point.index.tolist())]

            if len(direct_eval_point) > 0:
                direct_orch_info_point = self._concat_orch_data(df=direct_eval_point)
                direct_point_metrics   = self._execute_parallel(df_info      = direct_orch_info_point,
                                                                direct_eval  = direct_eval_point,
                                                                use_eval_set = True)
            else:
                direct_orch_info_point = None
                direct_point_metrics   = None

            if len(failed_eval_point) > 0:
                crude_orch_info_point = self._build_orchestration_crude(failed_eval = failed_eval_point,
                                                                        feat_type   = 'point')
                crude_point_metrics   = self._execute_parallel_crude(df_info      = crude_orch_info_point,
                                                                     use_eval_set = True)

            else:
                crude_orch_info_point = None
                crude_point_metrics   = None

            return [inf_point_orch, direct_eval_point, failed_eval_point, direct_orch_info_point, direct_point_metrics,
                    crude_orch_info_point, crude_point_metrics]

        else:
            return None

    def _line_eval(self) -> Union[List, None]:
        # Lines Performer Data - if applicable (4 hrs. 0 min. 52 sec.)
        if self.inf_line_schema is not None:
            inf_line_orch    = self._build_orchestration_match(inf_schema=self.inf_line_schema)
            direct_eval_line = inf_line_orch.query(self.direct_query)
            failed_eval_line = inf_line_orch.loc[inf_line_orch.index.difference(direct_eval_line.index.tolist())]

            if len(direct_eval_line) > 0:
                direct_orch_info_line = self._concat_orch_data(df=direct_eval_line)
                direct_line_metrics   = self._execute_parallel(df_info      = direct_orch_info_line,
                                                               direct_eval  = direct_eval_line,
                                                               use_eval_set = True)
            else:
                direct_orch_info_line = None
                direct_line_metrics   = None

            if len(self.failed_eval_line) > 0:
                crude_orch_info_line = self._build_orchestration_crude(failed_eval = failed_eval_line,
                                                                       feat_type   = 'line')
                crude_line_metrics  = self._execute_parallel_crude(df_info      = crude_orch_info_line,
                                                                   use_eval_set = True)

            else:
                crude_orch_info_line = None
                crude_line_metrics   = None

            return [inf_line_orch, direct_eval_line, failed_eval_line, direct_orch_info_line, direct_line_metrics,
                    crude_orch_info_line, crude_line_metrics]

        else:
            return None

    def _read_data(self, df: Union[str, None, DataFrame], to_concat: bool = False) -> Union[DataFrame, None]:
        """
        Read Parquet files if required.

        :param df: Dataset if applicable as string (parquet file) or DataFrame; otherwise, None.
        :param to_concat: Applicable only to match and crude schema files by concatenating parquet files containing
                          ground-truth information.
                            - id
                            - binary_value
                            - geo_value_vector
                            - matched_ratio
                            - count
                            - match_binary
                            - partition_file

        :return: DataFrame if applicable; otherwise, None.
        """
        if df is not None:
            if isinstance(df, str):
                if os.path.splitext(df)[-1] == ".parquet":
                    df = read_parquet(df)
                else:
                    df = read_csv(df)

            if to_concat:
                df = concat([read_parquet(d) for d in tqdm(df['geom_size_file'])])

            return df

        else:
            return None

    def _build_orchestra(self, grp_df: DataFrame, update_grnd_schema: DataFrame):
        synops_df   = read_parquet(grp_df['output_synopsis'].iloc[0])
        ngmdb_id    = grp_df['id'].iloc[0]

        if len(synops_df) > 0:
            # Used to identify geologic values associated to that ID
            sub_schema = update_grnd_schema.query('id == @ngmdb_id')

            # Enrich the ground-truth schema information with the performer information
            # Determine which values have been pre-successfully matched & which ones are missing.
            # Missing ones - will either be tossed out or run at later pipeline
            values_df  = (
                sub_schema
                .merge(synops_df, left_on=['geo_value_vector'], right_on=['geo_value'], suffixes=('_grnd', '_inf'))
                .assign(ratio           = lambda d: d['count_inf'] / d['count_grnd'],
                        combinations    = lambda d: d['count_grnd'] * d['count_inf']) # performers count per label / ground-truth count
            )

            # Identify which geologic values are missing from the performers
            missing_val = (
                sub_schema[~sub_schema['geo_value_vector'].isin(values_df['geo_value_vector'].unique())]
                ['geo_value_vector']
                .tolist()
            )

            num_missing  = len(missing_val) # Number of missing geologic values.
            total_values = len(sub_schema) # Total values from the ground-truth information.
            perc_missing = round(num_missing / total_values, 3) * 100 # Percentage missing geologic values
            valid_synops = True # Synopsis dataset is valid

        else:
            valid_synops = False
            missing_val  = None
            num_missing  = None
            perc_missing = 100
            values_df    = None

        update_df = grp_df.assign(valid_synops = valid_synops,
                                  missing_val  = [missing_val],
                                  num_missing  = num_missing,
                                  perc_missing = perc_missing,
                                  orch_info    = [values_df])

        return update_df

    def _build_orchestration_match(self, inf_schema: DataFrame) -> DataFrame:
        """
        Construct an orchestrated dataset that connects ground-truth information with performer information. It will
        be used during the main evaluation pipeline and for tracking / supplemental purposes.

        :param inf_schema: Performer information dataset.

        :return: Orchestrated DataFrame containing the following fields:
            - id                        => NGMDB ID
            - cog_id                    => COG ID linked to NGMDB ID
            - system                    => Performer system
            - system_version            => Performer system version
            - feature_type              => Feature Type (polygon, line, or point)
            - message                   => Download message from the CDR (should only be: "success-done")
            - data                      => Extraction message (should only be: "extracted data")
            - georeferenced_data        => Determines whether the performer data has been georeferenced or not (True, False) - part of eval process.
            - legend_data_added         => Determines whether legend data has been incorporated in the performer data (True, False) - part of eval process.
            - output_file               => If valid, location of the converted performer data as a GeoParquet file - part of eval process.
            - legend_extracted          => Location of the legend extracted related to performer data as a Parquet file.
            - output_synopsis           => If valid, location of the Parquet file that provides a synopsis of the performer data.
            - valid_synops              => Use this before eval process, flags which performer data succeeded and which didn't.
                                            True or False.
            - missing_val               => Use this before eval process to filter via not None. The valid ones would be a
                                           list of geologic values missing (if applicable / not empty) from the performers.
                                           For tracking.
            - num_missing               => Use this before eval process to filter via not None. The valid ones imply
                                           how many geologic values are missing vai missing_val field. Having 0 is ideal.
            - perc_missing              => Percentage of unique geologic values missing to assess in the ground-truth
                                           set from the performers.

            - orch_info                 => If valid from valid_synops & check for empty DataFrame, a DataFrame that
                                           combines from the synops and ground-truth.
                (From ground-truth)
                - id                    => NGMDB ID.
                - binary_value          => Geologic value from the binary rasters naming convention.
                - geo_value_vector      => Geologic value from the ground-truth set that matches to the binary_value.
                - matched_ratio_grnd    => Similarity ratio between geo_value_vector and binary_value.
                - count_grnd            => Number of geometries that exist in the ground-truth for that geologic value.
                                           To be used for data skewness reduction per CPU during eval.
                - match_binary_grnd     => True indicates that the matching process is direct via binary.
                - partition_file        => To be used during eval process, output of the partitioned ground-truth GeoParquet file for that geologic value.

                (From synopsis performer)
                - abbreviation          => What was abbreviated of the geologic value from the performer.
                - geo_value             => Geologic value that matches to the abbreviation (to be used during eval) from
                                           either annotated legend item IoU or similarity match binary value data.
                                           Invalid / undetermined would be "Use Spatial Index".
                - count_inf             => Number of geometries that the performer predicted for that geologic value.
                                           To be used for data skewness reduction per CPU during eval.
                - matched_ratio_inf     => Similarity matching
                                            - for polygons should be "None" as it is based on legend IoU.
                                            - for points & lines should be a number; if invalid or require "Use Spatial Index", then None.
                - match_binary_inf      => For polygons - None for direct matching as it is based on legend IoU;
                                           For points & lines - True (for performer based on matching binary value set).
                                                              - False (matched via crude value set or no matches and
                                                                       requires spatial indexing).
                - perc                  => Percentage of the performer's inferenced geometries for that geologic value
                                           representing their entire dataset.
                - ratio                 => Number of inferenced performer geometries over number of ground-truth geometries
                                           per geologic value.
                - combinations          => Potential number of combinations to (naively - assuming) assess based on the
                                           number of ground-truth geometries multiplied by number of performer geometries.

            - size_df                   => Size of the DataFrame via orch_info field. If applicable, otherwise 0 used to filter out.
            - path                      => Path to a TIF file when calculating F1 score.
        """

        """
        inf_schema => Retain only files from performers that were successfully downloaded & extracted.
                      - Use this to identify which IDs are not factored in & why.

        pre_match    => Supplement performer information with ground-truth information based on ID.
        """
        inf_schema = inf_schema.query('message == "success-done" and data == "extracted data"')
        pre_match  = inf_schema.merge(self.grnd_match_schema, on=['id'])

        # Re-organize performer information
        pre_inf_orch = (
            pre_match
            .groupby(self.unique_cols_grp, as_index=False)
            .agg({'output_file'      : lambda x: list(set(x.tolist())),
                  'legend_extracted' : lambda x: list(set(x.tolist())),
                  'output_synopsis'  : lambda x: list(set(x.tolist()))})
            .explode(['output_file', 'legend_extracted', 'output_synopsis']) # Assumes (tested) that there is only one each
        )

        # How many IDs that need to be assessed & update the ground-truth schema information
        ids_to_assess      = pre_inf_orch['id'].unique()
        update_grnd_schema = self.grnd_match_schema.query('id in @ids_to_assess').reset_index().drop(columns=['index'])

        tqdm.pandas()
        size_df    = lambda x: len(x) if isinstance(x, DataFrame) else 0
        match_orch = (
            pre_inf_orch
            .groupby(self.unique_cols_grp, as_index=False)
            .progress_apply(lambda a: concat([self._build_orchestra(grp_df=a, update_grnd_schema=update_grnd_schema)]))
            .assign(size_df = lambda d: list(map(size_df, d['orch_info'])))
            .reset_index()
            .drop(columns=['level_0', 'level_1'])
            .merge(self.feat_tif_file, on=['id'])
        )

        return match_orch

    def _build_crude(self, df: DataFrame, geo_field: str = Union['output_file', 'geom_size_file'], data_type: str = Union['grnd', 'inf']):
        if geo_field == "geom_size_file":
            pre_gdf   = read_parquet(df[geo_field].iloc[0])
            gdf       = gpq(pre_gdf['partition_file'].iloc[0])
            geom_type = gdf['geom_type'].iloc[0]

        else:
            gdf       = gpq(df[geo_field].iloc[0])
            geom_type = df['feature_type'].iloc[0]

        return DataFrame({'id'                 : [df['id'].iloc[0]],
                          'cog_id'             : [df['cog_id'].iloc[0]],
                          geo_field            : [df[geo_field].iloc[0]],
                          'geom_type'          : [geom_type],
                          f'count_{data_type}' : [len(gdf)]})

    def _build_orchestration_crude(self, failed_eval: DataFrame, feat_type: str = Union['point', 'line', 'polygon']) -> DataFrame:
        indirect_query = 'valid_synops == True and missing_val.notna() and num_missing.notna() and perc_missing == 100'
        pre_crude_set  = failed_eval.query(indirect_query).merge(self.grnd_crude_schema, on=['id'])

        tqdm.pandas()

        # Pre-orchestrate the ground-truth crude set
        pre_orch_grnd  = (
            pre_crude_set
            .groupby(['id', 'cog_id', 'geom_size_file'], as_index=False)
            .progress_apply(lambda a: concat([self._build_crude(df=a, geo_field='geom_size_file', data_type='grnd')]))
        )

        # Pre-orchestrate the performer crude set
        pre_orch_inf = (
            pre_crude_set
            .groupby(['id', 'cog_id', 'output_file'], as_index=False)
            .progress_apply(lambda a: concat([self._build_crude(df=a, geo_field='output_file', data_type='inf')]))
        )

        crude_orch_fin = (
            pre_orch_grnd
            .merge(pre_orch_inf, on=['id', 'cog_id'], suffixes=('_grnd', '_inf'))
            .assign(combinations = lambda d: d['count_grnd'] * d['count_inf'])
            .merge(pre_crude_set, on=['id', 'cog_id', 'geom_size_file', 'output_file'])
            .query('geom_type_grnd == @feat_type')
            [['id', 'cog_id', 'system', 'system_version', 'data', 'message', 'feature_type',
              'output_file', 'geom_size_file', 'path', 'count_grnd', 'count_inf', 'combinations']]
            .reset_index()
            .drop(columns=['index'])
            .reset_index()
            .rename(columns={'index' : 'ref_index'})
        )

        return crude_orch_fin

    def _concat_orch_data(self, df: DataFrame) -> DataFrame:
        orch_info_concat = (
            concat([o.orch_info.assign(ref_index=o.Index) for o in df.itertuples(index=True)])
            .reset_index()
            .drop(columns=['index'])
        )

        return orch_info_concat

    def _execute_parallel(self, df_info: DataFrame, direct_eval: DataFrame, use_eval_set: bool):
        L1        = Manager().list()
        equit_dfs = EquitDfs(df=df_info, geom_length='combinations').index_split
        keep_cols = ['id', 'cog_id', 'system', 'system_version', 'feature_type',
                     'message', 'data', 'georeferenced_data',
                     'legend_data_added', 'output_file', 'path']

        critical_data = [[c, direct_eval.loc[c['ref_index'].tolist()].drop_duplicates(keep_cols), d]
                         for d,c in enumerate(equit_dfs)]

        partial_func = partial(self._eval_pipeline, L1=L1, use_eval_set=use_eval_set, match_type='matched')
        ParallelPool(start_method='spawn', partial_func=partial_func, main_list=critical_data)
        concat_df = concat(L1)

        return concat_df

    def _execute_parallel_crude(self, df_info: DataFrame, use_eval_set: bool):
        L1            = Manager().list()
        #equit_dfs     = EquitDfs(df=df_info, geom_length='combinations').index_split
        equit_dfs     = np.array_split(df_info, cpu_count())
        critical_data = [[c, d] for d,c in enumerate(equit_dfs)]
        partial_func  = partial(self._eval_pipeline, L1=L1, use_eval_set=use_eval_set, match_type='crude')
        ParallelPool(start_method='spawn', partial_func=partial_func, main_list=critical_data)
        concat_df     = concat(L1)

        return concat_df

    def _h3_expansion(self, g_df: GeoDataFrame, p_df: GeoDataFrame) -> List:
        """
        Expand H3 index values of both ground and performer data and use the unique index values from the ground truth
        to filter performer data before proceeding metrics evaluation.

        :param g_df: Ground-truth data.
        :param p_df: Performer data.

        :return: List --> Expanded ground-truth dataset via H3 field and filtered performer's dataset based on minimum H3 index.
        """
        # Expand min H3 and max H3 to acquire matching values
        g_df_exp = g_df.explode('min_h3').explode('max_h3').reset_index()
        p_df_exp = p_df.explode('min_h3').explode('max_h3')

        # Unique min and max H3 values from the ground-truth - use that to filter out the performer's features
        uniq_min_h3 = g_df_exp['min_h3'].unique()
        uniq_max_h3 = g_df_exp['max_h3'].unique()

        # Filtered performer's dataset based on minimum H3 index
        p_df_sub_exp = p_df_exp.query('min_h3 in @uniq_min_h3').query('max_h3 in @uniq_max_h3')

        return [g_df_exp, p_df_sub_exp]

    def _filter_by_h3(self, p_df_sub_exp, p_df, g_df_exp, g_df) -> List:
        """
        Finalize filter of the performer set based on unique index values from filtered H3 values (min and max) and final filter
        of the ground-truth set based on the unique index values from filtered H3 values (min and max).

        :param p_df_sub_exp: Filtered and expanded performer dataset.
        :param p_df: Original performer dataset that will be filtered and finalized for evaluation metrics.
        :param g_df_exp: Filtered and expanded ground-truth dataset.
        :param g_df: Original ground-truth dataset that will be filtered and finalized for evaluation metrics.

        :return: List containing the finalized filtered performer and ground-truth set for evaluation metrics.
        """

        get_unique_idx = p_df_sub_exp['index'].unique()
        filt_p_df      = p_df.query('index in @get_unique_idx')
        u_min_h3       = p_df_sub_exp['min_h3'].unique()
        u_max_h3       = p_df_sub_exp['max_h3'].unique()
        filt_g_df_exp  = g_df_exp.query('min_h3 in @u_min_h3').query('max_h3 in @u_max_h3')
        filt_g_df      = g_df.iloc[filt_g_df_exp['index'].unique(), :]

        del filt_g_df_exp, get_unique_idx, u_min_h3

        return [filt_p_df, filt_g_df]

    def _morton_index_array(self, g_array: np.array):
        z_func = lambda x, y: SpatialOps().morton_z(x=x, y=y)

        g_df = (
            DataFrame(g_array, columns=['x', 'y'])
            .assign(z_index = lambda d: list(map(z_func, d['x'], d['y'])))
            .sort_values('z_index')
            #.set_index('z_index')
        )

        return g_df

    def _pre_f1(self, geom, feat_type):
        if feat_type == "polygon":
            p_geom  = Polygon([SpatialOps().coord2pixel_raster(raster_transform=self.transform, xs=p[0], ys=p[1])
                               for p in geom.exterior.coords])
            p_array = SpatialOps().get_pixel_coordinates(polygon=p_geom, img_dim=self.img_dim)

        elif feat_type == "line":
            p_geom  = LineString([SpatialOps().coord2pixel_raster(raster_transform=self.transform, xs=p[0], ys=p[1])
                                  for p in geom.coords])
            p_array = SpatialOps().get_pixel_coordinates_by_line(line=p_geom, img_dim=self.img_dim)

        elif feat_type == "point":
            p_array = np.array([SpatialOps().coord2pixel_raster(raster_transform=self.transform, xs=geom.x, ys=geom.y)])

        return p_array

    def _f1(self, pdf, indices, g_array):

        # Localized F1
        try:
            pre_f1    = self._morton_index_array(g_array=g_array)

            try:
                len_pdf        = len(pdf)
                len_gdf        = len(pre_f1)
                true_positives = len(pre_f1.query('z_index in @indices'))

                del pre_f1

                message = "success-f1"

                try:
                    precision = true_positives / len_pdf
                except ZeroDivisionError:
                    precision = 0
                    len_pdf   = 0
                    message   = "zero-p_df"

                try:
                    recall = true_positives / len_gdf
                    f1     = (2 * precision * recall) / (precision + recall)

                    return [len_pdf, len_gdf, true_positives, precision, recall, f1, message]

                except ZeroDivisionError:
                    recall  = 0
                    len_gdf = 0
                    f1      = 0

                    if message == "zero-p_df":
                        message = "zero-p_df & g_df"
                    else:
                        message = "zero-g_df"

                    return [len_pdf, len_gdf, true_positives, precision, recall, f1, message]

            except KeyError:
                message = "key-error"
                return [None, None, 0, 0, 0, None, message]

        except ValueError:
            message = 'value-error'
            return [None, None, 0, 0, 0, None, message]

    def _acq_metrics(self, filt_p_df, filt_g_df, geo_value, geo_value_query: bool, data_source):
        feat_type    = self.feat_type
        pre_exp_g_df = filt_g_df.explode('min_h3').explode('max_h3')
        z_func       = lambda x, y: SpatialOps().morton_z(x=x, y=y)

        pre_eval     = []
        # Iterate through performer candidates
        for p in tqdm(filt_p_df.itertuples(index=True), total=filt_p_df.shape[0]):
            tmp_p_geom    = p.geom
            tmp_p_index   = p.index
            tmp_p_h3_min  = p.min_h3
            tmp_p_h3_max  = p.max_h3

            if self.feat_type == "line" or self.feat_type == "polygon":
                sub_filt_exp_g_df = (
                    pre_exp_g_df
                    .query('min_h3 in @tmp_p_h3_min')
                    .query('max_h3 in @tmp_p_h3_max')
                )

            # since it is 1D and does not traverse - will need to keep at a more general spatial index level.
            # a bit slower than filtering down to more refine spatial index level, but is more accurate.
            elif self.feat_type == "point":
                sub_filt_exp_g_df = (
                    pre_exp_g_df
                    .query('min_h3 in @tmp_p_h3_min')
                )

            if len(sub_filt_exp_g_df) > 0:

                unique_indexes = list(set(sub_filt_exp_g_df.index.to_list()))
                sub_filt_g_df  = filt_g_df.loc[unique_indexes, :]

                del sub_filt_exp_g_df

                # Iterate through filtered ground-truth
                tmp_data = []
                for g in sub_filt_g_df.itertuples(index=True):
                    tmp_g_geom    = g.geom

                    if self.use_eval_set and self.feat_type == "point" or self.feat_type == "line":
                        acq_map_scale = (
                            self.eval_set[(self.eval_set[self.eval_id_field] == self.performer['id']) &
                                          (self.eval_set[self.eval_cog_id_field] == self.performer['cog_id'])]
                            [self.eval_scale_field]
                            .iloc[0]
                        )

                        # dynamic buffer
                        dyna_buffer = (acq_map_scale * self.dynamic_buffer) / SpatialOps().map_scale_inch2meter
                        tmp_p_geom  = tmp_p_geom.buffer(dyna_buffer)
                        tmp_g_geom  = tmp_g_geom.buffer(dyna_buffer)
                        feat_type   = "polygon"

                    # Proceed to metrics if there is a Spatial overlap
                    if overlaps(tmp_p_geom, tmp_g_geom):

                        try:
                            iou = SpatialOps().IoU_calc(geom1=tmp_g_geom, geom2=tmp_p_geom)
                        except ZeroDivisionError:
                            iou = 0

                        tmp_data.append([self.performer['id'], self.performer['cog_id'], self.performer['message'],
                                         self.performer['data'], self.performer['system'], self.performer['system_version'],
                                         self.feat_type, feat_type, geo_value, tmp_p_index, g.Index, data_source, iou,
                                         self.pix_data, self.performer['path'],
                                         geo_value_query, len(filt_p_df), len(sub_filt_g_df),
                                         self.grnd_count, self.inf_count, True, True, tmp_p_geom, tmp_g_geom])

                    else:
                        tmp_data.append([self.performer['id'], self.performer['cog_id'], self.performer['message'],
                                         self.performer['data'], self.performer['system'], self.performer['system_version'],
                                         self.feat_type, feat_type, geo_value, tmp_p_index, g.Index, data_source, 0,
                                         self.pix_data, self.performer['path'], geo_value_query, len(filt_p_df),
                                         len(sub_filt_g_df), self.grnd_count, self.inf_count, True, False, tmp_p_geom, tmp_g_geom])

                if len(tmp_data) > 0:
                    tmp_df  = DataFrame(tmp_data, columns=self.important_cols)
                    max_iou = tmp_df['iou'].max()
                    tmp_df  = tmp_df.query('iou == @max_iou and success_overlap == True')
                    # pre_eval.append(tmp_df)

                    if len(tmp_df) > 0:
                        p_array   = self._pre_f1(geom=tmp_df['perf_geom'].iloc[0], feat_type=tmp_df['mod_feat_type'].iloc[0])
                        p_df      = DataFrame(p_array, columns=['x', 'y']).assign(z_index=lambda d: list(map(z_func, d['x'], d['y'])))
                        p_indices = sorted(p_df['z_index'].unique())
                        del p_array
                        get_all_F1 = [[t, self._f1(pdf     = p_df,
                                                   indices = p_indices,
                                                   g_array = self._pre_f1(geom=tmp_df['grnd_geom'].iloc[t],
                                                                          feat_type=tmp_df['mod_feat_type'].iloc[t]))]
                                      for t in range(len(tmp_df))]

                        pre_F1 = DataFrame(get_all_F1, columns=['loc', 'data'])
                        pre_F1[['len_pdf', 'len_gdf', 'true_positives', 'precision', 'recall', 'f1', 'f1_message']] = pre_F1['data'].apply(Series)
                        max_F1 = pre_F1['f1'].max()
                        try:
                            tmp_F1 = pre_F1.query('f1 == @max_F1').iloc[0]
                            fin_df = DataFrame(tmp_df.iloc[tmp_F1['loc'], :-2]).transpose()
                            fin_df = fin_df.assign(len_pdf = tmp_F1['len_pdf'],
                                                   len_gdf = tmp_F1['len_gdf'],
                                                   true_positives = tmp_F1['true_positives'],
                                                   precision      = tmp_F1['precision'],
                                                   recall         = tmp_F1['recall'],
                                                   f1             = tmp_F1['f1'],
                                                   f1_message     = tmp_F1['f1_message'])

                            pre_eval.append(fin_df)

                        except IndexError:
                            pre_F1 = pre_F1.drop(columns=['loc', 'data'])
                            fin_df = concat([tmp_df.iloc[:, :-2], pre_F1], axis=1)
                            pre_eval.append(fin_df)

        if len(pre_eval) > 0:
            warnings.filterwarnings(action='ignore', category=FutureWarning)
            concat_df = concat(pre_eval)
            return concat_df

        else:
            return None

    def _metrics_pipeline(self, grp_orch_info: DataFrame, df_eval_inf: Union[DataFrame, None], partition_value, match_type):

        # Acquire appropriate performer data to be assessed & read its output file
        ref_index = grp_orch_info['ref_index'].iloc[0]  # possibly use this in subsequent process to join back

        if match_type == "matched":
            self.performer = df_eval_inf.loc[ref_index]

        elif match_type == "crude":
            self.performer = grp_orch_info.loc[ref_index]

        perform_data   = gpq(self.performer['output_file'])
        self.feat_type = self.performer['feature_type']

        # Raster characteristics to be used for F1
        raster_tif_g   = SpatialOps().raster_characteristics(tif_file=self.performer['path'])
        tif            = raster_tif_g[0].read(1)
        self.img_dim   = (raster_tif_g[-1], raster_tif_g[-2]) # Height and Width
        self.transform = raster_tif_g[2]
        self.pix_data  = len(np.where(tif == 1)[0])

        del tif

        if match_type == "matched":

            tmp_info = []
            for g in grp_orch_info.itertuples(index=True):
                grnd_data       = gpq(g.partition_file)
                geo_value       = g.geo_value
                self.grnd_count = g.count_grnd
                self.inf_count  = g.count_inf

                # Filter by geologic value - might need to put an if statement such as if it is a direct match or not
                # Followed by expansion of H3 spatial index
                perform_sub  = perform_data.query('geo_value == @geo_value')
                p_exp_h3     = self._h3_expansion(g_df=grnd_data, p_df=perform_sub)
                p_df_sub_exp = p_exp_h3[1]
                g_df_exp     = p_exp_h3[0]

                del p_exp_h3

                # If the filtered expanded performer dataset has data
                if len(p_df_sub_exp) > 0:
                    pre_process = self._filter_by_h3(p_df_sub_exp = p_df_sub_exp,
                                                     g_df_exp     = g_df_exp,
                                                     p_df         = perform_sub,
                                                     g_df         = grnd_data)

                    filt_p_df = pre_process[0]
                    filt_g_df = pre_process[1]

                    del p_df_sub_exp, g_df_exp

                    pre_fin_df = self._acq_metrics(filt_p_df       = filt_p_df,
                                                   filt_g_df       = filt_g_df,
                                                   geo_value       = geo_value,
                                                   geo_value_query = True,
                                                   data_source     = g.partition_file)

                    # Append results
                    tmp_info.append(pre_fin_df)

                # Handling unknown geologic values - need to do further cleaning after this process to then drop other candidate & calculated values
                unknown_sub = perform_data.query('geo_value == "Use Spatial Index"')
                if len(unknown_sub) > 0:
                    p_exp_h3     = self._h3_expansion(g_df=grnd_data, p_df=perform_sub)
                    p_df_sub_exp = p_exp_h3[1]
                    g_df_exp     = p_exp_h3[0]

                    del p_exp_h3

                    # If the filtered expanded performer dataset has data
                    if len(p_df_sub_exp) > 0:
                        pre_process = self._filter_by_h3(p_df_sub_exp = p_df_sub_exp,
                                                         g_df_exp     = g_df_exp,
                                                         p_df         = perform_sub,
                                                         g_df         = grnd_data)

                        filt_p_df = pre_process[0]
                        filt_g_df = pre_process[1]

                        del p_df_sub_exp, g_df_exp

                        pre_fin_df = self._acq_metrics(filt_p_df       = filt_p_df,
                                                       filt_g_df       = filt_g_df,
                                                       geo_value       = geo_value,
                                                       geo_value_query = False,
                                                       data_source     = g.partition_file)

                        # Append results
                        tmp_info.append(pre_fin_df)

            if len(tmp_info) > 0:
                concat_df = concat(tmp_info)
                #print(concat_df, concat_df.dtypes)
                main_dir  = f"{self.output_dir}/{self.feat_type}"
                DataEng.checkdir(dir_name=main_dir)
                file_path = f"{main_dir}/{self.feat_type}_partition{partition_value}_{ref_index}_{match_type}.parquet"
                #print(file_path)
                concat_df.to_parquet(path=file_path)

                return concat_df

        elif match_type == "crude":
            tmp_info = []
            for g in grp_orch_info.itertuples(index=True):
                pre_grnd_data   = read_parquet(g.geom_size_file)
                partition_file  = pre_grnd_data['partition_file'].iloc[0]
                grnd_data       = gpq(partition_file)
                geo_value       = None
                self.grnd_count = g.count_grnd
                self.inf_count  = g.count_inf

                p_exp_h3     = self._h3_expansion(g_df=grnd_data, p_df=perform_data)
                p_df_sub_exp = p_exp_h3[1]
                g_df_exp     = p_exp_h3[0]

                del p_exp_h3, pre_grnd_data

                if len(p_df_sub_exp) > 0:
                    pre_process = self._filter_by_h3(p_df_sub_exp=p_df_sub_exp,
                                                     g_df_exp=g_df_exp,
                                                     p_df=perform_data,
                                                     g_df=grnd_data)

                    filt_p_df = pre_process[0]
                    filt_g_df = pre_process[1]

                    del p_df_sub_exp, g_df_exp

                    pre_fin_df = self._acq_metrics(filt_p_df=filt_p_df,
                                                   filt_g_df=filt_g_df,
                                                   geo_value=geo_value,
                                                   geo_value_query=True, # should be set to False
                                                   data_source = partition_file)

                    # Append results
                    tmp_info.append(pre_fin_df)

            if len(tmp_info) > 0:
                concat_df = concat(tmp_info)
                main_dir  = f"{self.output_dir}/{self.feat_type}"
                DataEng.checkdir(dir_name=main_dir)
                file_path = f"{main_dir}/{self.feat_type}_partition{partition_value}_{ref_index}_{match_type}.parquet"
                #print(file_path)
                concat_df.to_parquet(path=file_path)

                return concat_df

    def _eval_pipeline(self, critical_data: List, use_eval_set: bool, L1, match_type = Union['matched', 'crude']):

        tqdm.pandas()

        self.use_eval_set   = use_eval_set
        self.important_cols = ['id', 'cog_id', 'performer_message', 'data', 'system', 'system_version',
                               'feat_type', 'mod_feat_type', 'geo_value_vector', 'p_index', 'g_index', 'grnd_file', 'iou',
                               'total_pix', 'tif_file', 'geo_value_query', 'num_cand_perf', 'num_cand_geom',
                               'grnd_count', 'inf_count', 'filtered_H3', 'success_overlap', 'perf_geom', 'grnd_geom']


        if match_type == "matched":
            orch_info_df        = critical_data[0]
            df_eval_inf         = critical_data[1]
            partition_value     = critical_data[2]

            try:
                orch_grp = (
                    orch_info_df
                    .groupby(['ref_index'], as_index=False)
                    .progress_apply(lambda a: concat([self._metrics_pipeline(grp_orch_info=a,
                                                                             df_eval_inf=df_eval_inf,
                                                                             partition_value=partition_value,
                                                                             match_type='matched')]))
                )

                if len(orch_grp) > 0:
                    L1.append(orch_grp)

            except ValueError:
                pass

        elif match_type == "crude":
            orch_info_df    = critical_data[0]
            partition_value = critical_data[1]

            try:
                orch_grp = (
                    orch_info_df
                    .groupby(['ref_index'], as_index=False)
                    .progress_apply(lambda a: concat([self._metrics_pipeline(grp_orch_info=a,
                                                                             df_eval_inf=None,
                                                                             partition_value=partition_value,
                                                                             match_type='crude')]))
                )

                if len(orch_grp) > 0:
                    L1.append(orch_grp)

            except ValueError:
                pass
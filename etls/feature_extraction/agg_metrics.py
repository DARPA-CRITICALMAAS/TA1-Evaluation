from tqdm import tqdm
from pandas import concat, qcut, read_parquet, read_csv, DataFrame, errors
import warnings

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ..utils import discover_docs


class AggMetrics:

    def __init__(self, polygon_results_path, line_results_path, point_results_path, eval_set, q_decile: int = 10):
        self.q_decile    = q_decile
        self.eval_set    = read_csv(eval_set)

        self.grp_geo_map = ['id', 'cog_id', 'performer_message',
                            'data', 'system', 'system_version',
                            'geo_value_vector', 'f1_message']
        self.grp_map     = ['id', 'cog_id', 'performer_message',
                            'data', 'system', 'system_version']
        self.grp_perf    = ['performer_message', 'data', 'system', 'system_version']
        self.decile_grp  = ['id', 'cog_id', 'performer_message',
                            'data', 'system', 'system_version']

        self.lambda_rename = lambda x: "_".join(x)

        self.polygon_df = self._concat_data(data_set=discover_docs(path=polygon_results_path))
        self.line_df    = self._concat_data(data_set=discover_docs(path=line_results_path))
        self.point_df   = self._concat_data(data_set=discover_docs(path=point_results_path))

        self.polygon_agg_stats = self._general_stats(df=self.polygon_df)
        self.line_agg_stats    = self._general_stats(df=self.line_df)
        self.point_agg_stats   = self._general_stats(df=self.point_df)

    def _agg_decile(self, df: DataFrame, metric_type: str, agg_dict):
        agg_dec_df = (
            df
            .pipe(lambda a: a.assign(**{f'{metric_type}_median_q' : a[metric_type]}))
            .groupby(['performer_message', 'data', 'system', 'system_version'], as_index=False)
            .apply(self._qcut_decile, cols=[f'{metric_type}_median_q'])
            .groupby(self.grp_perf + [f'{metric_type}_median_q'], as_index=False)
            .agg(agg_dict)
        )

        agg_dec_df.columns = self.grp_perf + list(map(self.lambda_rename, agg_dec_df.columns[len(self.grp_perf):]))

        return agg_dec_df

    def _qcut_decile(self, grp, cols):
        for col in cols:
            grp[col] = qcut(grp[col], q=self.q_decile, labels=False, duplicates='drop')
        return grp

    def _stats_process(self, df, agg_dict, grp_cols):
        warnings.filterwarnings(action='ignore', category=errors.PerformanceWarning)
        drop_cols = ['ID', 'COG ID']

        grp_df = (
            df
            .groupby(grp_cols, as_index=False)
            .agg(agg_dict)
            .reset_index()
            .drop(columns=['index'])
        )

        grp_df.columns = grp_cols + list(map(self.lambda_rename, grp_df.columns[len(grp_cols):]))
        grp_df = grp_df.merge(self.eval_set, left_on=['id', 'cog_id'], right_on=['ID', 'COG ID']).drop(columns=drop_cols)

        return grp_df

    def _general_stats(self, df):
        df = df.assign(candidate_ratio = lambda d: d['num_cand_perf'] / d['num_cand_geom'],
                       inf_grnd_ratio  = lambda d: d['inf_count'] / d['grnd_count'])

        # groupby performer, map, and geologic value
        map_geo_agg  = {'iou'             : ['mean', 'median', 'count'],
                        'f1'              : ['mean', 'median'],
                        'candidate_ratio' : ['mean', 'median'],
                        'inf_grnd_ratio'  : ['mean', 'median']}
        perf_map_geo   = self._stats_process(df=df, agg_dict = map_geo_agg, grp_cols=self.grp_geo_map)

        agg_dict = {'id'              : 'nunique',
                    'geo_value_vector': 'nunique',
                    'iou_count'       : 'sum',
                    'iou_median'      : ['mean', 'median'],
                    'f1_median'       : ['mean', 'median']}

        try:
            p_geo_iou_median   = self._agg_decile(df=perf_map_geo, metric_type='iou_median', agg_dict=agg_dict)
            p_geo_f1_median    = self._agg_decile(df=perf_map_geo, metric_type='f1_median', agg_dict=agg_dict)

        except ValueError:
            p_geo_iou_median = None
            p_geo_f1_median  = None

        # groupby performer and map
        map_agg = {'geo_value_vector': 'nunique',
                   'iou'             : ['mean', 'median', 'count'],
                   'f1'              : ['mean', 'median'],
                   'candidate_ratio' : ['mean', 'median'],
                   'inf_grnd_ratio'  : ['mean', 'median']}
        perf_map     = self._stats_process(df=df, agg_dict=map_agg, grp_cols=self.grp_map)
        agg_dict     = {'id'         : 'nunique',
                        'iou_count'  : 'sum',
                        'iou_median' : ['mean', 'median'],
                        'f1_median'  : ['mean', 'median']}
        try:
            p_map_iou_median = self._agg_decile(df=perf_map, metric_type='iou_median', agg_dict=agg_dict)
            p_map_f1_median  = self._agg_decile(df=perf_map, metric_type='f1_median', agg_dict=agg_dict)

        except ValueError:
            p_map_iou_median = None
            p_map_f1_median  = None

        overall_perf = perf_map.groupby(self.grp_perf, as_index=False).agg({'iou_median'             : ['mean', 'median'],
                                                                            'f1_median'              : ['mean', 'median'],
                                                                            'candidate_ratio_median' : ['mean', 'median'],
                                                                            'inf_grnd_ratio_median'  : ['mean', 'median']})

        overall_perf.columns = self.grp_perf + list(map(self.lambda_rename, overall_perf.columns[len(self.grp_perf):]))

        return perf_map_geo, p_geo_iou_median, p_geo_f1_median, perf_map, p_map_iou_median, p_map_f1_median, overall_perf

    def _concat_data(self, data_set):
        df = concat([read_parquet(t) for t in tqdm(data_set['path'])])
        return df

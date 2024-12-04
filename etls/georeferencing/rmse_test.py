"""

"""

import numpy as np
from pandas import concat, DataFrame
from typing import List

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ..utils import SpatialOps


class RMSE_NonGeodesic_Test:

    def __init__(self, df: DataFrame, base_val: int, grnd_pnt_field: str, groupby_fields: List = ["id", "tif_file"], geog_unit_field = "orig_geog_units"):
        # Assumes you are evaluating non-decimal degree numbers

        point_coord_func     = SpatialOps().lambda_points2coord()
        gen_random_pred_func = lambda x: SpatialOps().generate_random_points_from_pnt_list(pnt_value = x,
                                                                                           base_val  = base_val)

        keep_cols = ['id', 'tif_file', 'pix_height', 'pix_width', 'orig_epsgs', 'orig_geog_units',
                     'proj_epsgs', 'proj_geog_units', 'same_epsgs', 'grnd_pnts', 'pred_pnts', 'rmse']

        new_cols = ["unit_converter", "unit_matched", "unit_similarity"]

        self.rmse_per_map = (
            df
            # Groupby - usually by ID and tif_file & parse out the Shapely point coordinates into a nested NumPy array.
            # The reason is that to eventually determine what units are these coordinates in.
            .groupby(groupby_fields)
            .apply(lambda a: concat([DataFrame({"grnd_pnts" : [np.array(list(a[grnd_pnt_field].apply(point_coord_func)))]})]), include_groups=False)
            # Randomly generate predictive points into a nested NumPy array.
            .assign(pred_pnts = lambda a: a[['grnd_pnts']].apply(lambda e: np.array(list(map(gen_random_pred_func, *e))), axis=1))
            .reset_index()
            # Groupby - usually by ID and tif file & calculate the RMSE
            .groupby(groupby_fields)
            .apply(lambda a: a.assign(rmse = lambda d: d[['grnd_pnts', 'pred_pnts']].apply(lambda e: SpatialOps().non_geodesic_rmse(*e), axis=1)), include_groups=False)
            # Drop unnecessary columns
            .drop(columns=[f"level_{len(groupby_fields)}"])
            .reset_index()
            .drop(columns=[f"level_{len(groupby_fields)}"])
            # Supplement the DataFrame by merging with the map file since it contains columns filled with characteristics
            # Important to do proper conversion of the RMSE per map to 1 universal unit (e.g., metre or kilometer)
            .merge(df, on=groupby_fields, how='left')
            .drop_duplicates(groupby_fields + ['pix_height', 'pix_width'])
            [keep_cols]
            # Identify per geographic unit the matching conversion required.
            # WARNING: Some of the geographic units might be labelled a bit differently; therefore, using sequence matching
            # and receiving the maximum (i.e., most similar) value.
            .assign(pre_conv=lambda a: list(map(SpatialOps().acq_unit_converter_by_other_unit, a[geog_unit_field])))
            # Split the column containing list values into DataFrame and c-bind back to the merged-RMSE DataFrame
            .pipe(lambda a: concat([a.reset_index(), DataFrame(a['pre_conv'].to_list(), columns=new_cols).reset_index()], axis=1))
            # Conduct unit conversion to meters and re-convert to kilometers
            .assign(rmse_km=lambda a: (a['rmse'] * a['unit_converter']) / 1000)
            .drop(columns=['index', 'pre_conv'])
        )

        self.rmse_stats = (
            DataFrame(self.rmse_per_map['rmse_km'].describe()[['count', 'min', 'max', 'std', 'mean']])
            .transpose()
            .assign(median = self.rmse_per_map['rmse_km'].median())
        )
"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer.

Date Created: Sept. 2024.

Last Update: Sept. 2024.

About: Prepare evaluation for georeferencing by taking each TIF file from the evaluation set and randomly select
       n number of ground control points from the pixel space.

Output:
    DataFrame with the following schema:
        - id                => NGMDB ID / unique identifier of the map.
        - tif_file          => path to the TIF file.
        - pix_height        => Maximum height of the TIF file in pixels.
        - pix_width         => Maximum width of the TIF file in pixels.
        - random_pix_height => Random pixel value from the height (i.e., y-axis).
        - random_pix_width  => Random pixel value from the width (i.e., x-axis).
        - orig_epsgs        => original EPSG (spatial reference system).
        - orig_geog_units   => What unit is the geographic coordinate in.
        - orig_pnts         => Shapely point converted pixel space in orig_epsgs.
        - deci_pnts         => Shapely point from orig_pnts converted to decimal degrees via epsg:4326.
        - proj_epsgs        => Re-projected EPSG if required, otherwise it will be the same as orig_epsgs.
        - proj_geog_units   => What unit is the re-projected geographic coordinate in.
        - proj_pnts         => If reprojection is required, Shapely point re-projected from deci_pnts to its UTM zone.
                               Otherwise, it is the same as orig_pnts.
        - same_epsgs        => Quick way to determine if the Shapely point had to be reprojected or not based. If True,
                               then the proj_epsgs and proj_pnts are the same as orig_epsgs and orig_pnts respectively.
"""
import os.path
from typing import Union
from pandas import DataFrame, read_csv, concat, read_parquet
from etls.utils import ParallelPool, SpatialOps
from multiprocessing import Manager, cpu_count
from functools import partial
import numpy as np
from tqdm import tqdm

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ..utils import DataEng


class PreEvalGeoref:

    def __init__(self, tif_files: Union[DataFrame, str], eval_df: Union[DataFrame, str, dict], num_gcps: int = 10,
                 id_field: str = "id", eval_id_field: str = "ID", eval_cog_id_field: str = "COG ID",
                 file_path_field: str = "path", unique_samples: bool = False):
        """
        :param tif_files: List of tif files including their respective paths.
        :param num_gcps: Number of ground control points to select randomly. Default is 10.
        :param id_field: Provide the field that contains unique identifier of the maps. Default is "id".
        :param file_path_field: Provide the field that contains the paths to tifs. Default is "path".
        :parma unique_samples: A NumPy parameter. Boolean - whether to have unique samples randomly generated or not. Default is False which
                               means that it will have a random unique sample set. True will not.
        """

        eval_df     = self._eval_df(eval_df           = eval_df,
                                    eval_id_field     = eval_id_field,
                                    eval_cog_id_field = eval_cog_id_field)
        eval_lambda = lambda x: eval_df[x]

        if isinstance(tif_files, str):
            get_ext = os.path.splitext(tif_files)[-1]
            if get_ext == ".parquet":
                tif_files = read_parquet(tif_files)
            elif get_ext == ".csv":
                tif_files = read_csv(tif_files)

        self.num_gcps        = num_gcps
        self.unique_samples  = unique_samples
        self.file_path_field = file_path_field
        self.id_field        = id_field

        # Partition out the DataFrame, parallel-process to read TIF files, and generate random GCPs with spatial reference information.
        split_dfs    = np.array_split(tif_files, cpu_count())
        L1           = Manager().list()
        partial_func = partial(self._select_random_gcps, L1 = L1)
        ParallelPool(start_method="spawn", partial_func=partial_func, main_list=split_dfs)

        self.concat_df = (
            concat(L1)
            .reset_index()
            .drop(columns=['index'])
            .assign(cog_id = lambda a: list(map(eval_lambda, a['id'])))
        )

    def _select_random_gcps(self, list_of_tifs: DataFrame, L1):
        """
        Read each TIF file, acquire spatial reference system information, randomly select n number of ground control
        points, convert to spatial point, and re-project spatial point if necessary.

        :param list_of_tifs: Partitioned DataFrame from the list of TIF files.
        :param L1: List Manager to append results during parallel processing.
        """

        for row in tqdm(range(len(list_of_tifs))):
            # Row from the dataframe, acquire tif path, and ID of the map.
            tmp_row = list_of_tifs.iloc[row]
            tif     = tmp_row[self.file_path_field].replace("\\", "/")
            id_map  = tmp_row[self.id_field]

            # Construct DataFrame by randomly selecting pixel space, converting to spatial points, check spatial
            # referencing system, and re-project if necessary.
            tmp_df  = SpatialOps().select_random_gcps_from_tif(id_map   = id_map,
                                                               tif_file = tif)
            # Append dataframe to list manager
            L1.append(tmp_df)

    def _eval_df(self, eval_df, eval_id_field, eval_cog_id_field) -> dict:

        if isinstance(eval_df, str):
            get_ext = os.path.splitext(eval_df)[-1]

            if get_ext == ".json":
                eval_df = DataEng.read_json(config_file=eval_df)

            elif get_ext == ".csv":
                eval_df = read_csv(eval_df)

        if isinstance(eval_df, DataFrame):

            eval_df = eval_df[[eval_id_field, eval_cog_id_field]].drop_duplicates([eval_id_field, eval_cog_id_field])
            eval_df = {i: c for i, c in zip(eval_df[eval_id_field], eval_df[eval_cog_id_field])}

        return eval_df

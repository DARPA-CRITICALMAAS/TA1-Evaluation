"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer.

Date Created: Sept. 2024.

Last Update: Sept. 2024.

About: Quick inspection of the Georeferencing evaluation (i.e., ground-truth) set by looking at if the amount of IDs
       in the inventory is equal to the expected number of IDs and how many maps exist per ID. Typically, should be
       one map per ID; however, if there are more than 1 map per ID, check to make sure that they are sub-maps.
"""

from pandas import DataFrame, concat
from tqdm import tqdm
import warnings

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ..utils import discover_docs


class GeorefInspect:

    def __init__(self, georef_path: str, inhouse_inventory: DataFrame, expected_ids: int = 140):
        """
        :param georef_path: Path to where the Georeferencing evaluation set is stored.
        :param inhouse_inventory: Inventory list of the Georeferencing dataset.
        :param expected_ids: Expected number of unique IDs that should be in the inventory. Default is 140 unique IDs.
        """

        warnings.filterwarnings(action='ignore', category=FutureWarning)

        self.tif_files = (
            # Create DataFrame - append ID to the georef_path
            concat([discover_docs(path = f"{georef_path}/{i}").assign(id = i)
                    for i in tqdm(inhouse_inventory['id'])])
            # Assign file extension, query only that have TIF files, and keep significant columns.
            .assign(file_ext = lambda d: d[['filename']].apply(lambda e: os.path.splitext(*e)[-1], axis=1))
            .query('file_ext == ".tif"')
            [['id', 'directory', 'path', 'filename', 'file_ext']]
            # Drop duplicates based on ID and filename to minimize duplicates.
            .drop_duplicates(['id', 'filename'], keep='first')
        )

        # Groupby ID and count how many basemap TIF files there are. Ideally, should be 1 per ID. If there are multiple,
        # then likely there are multiple maps or sub-maps.
        self.tif_grp = (
            self.tif_files
            .groupby('id', as_index=False)
            .agg('count')
            [['id', 'directory']]
        )

        # Number of missing IDs
        self.total_missing = len(self.tif_grp) - expected_ids
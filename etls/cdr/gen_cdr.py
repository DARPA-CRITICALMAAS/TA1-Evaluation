# Miscellaneous packages
from typing import List, Union
from pandas import read_csv, DataFrame

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ..utils import DataEng, SpatialOps


class Generic:

    def __init__(self):
        pass

    @staticmethod
    def acquire_cog_ids_from_csv(cog_ids: Union[list, str], cog_id_field) -> List:
        """
        Acquire COG IDs either from CSV or list.

        :param cog_ids: COG IDs from CSV or list.
        :param cog_id_field: If CSV provided, then it will be a COG ID Field, otherwise default to None.

        :return: List of COG IDs.
        """
        if isinstance(cog_ids, str):
            try:
                cog_df = read_csv(cog_ids)
            except UnicodeDecodeError:
                cog_df = read_csv(cog_ids, encoding='ISO-8859-1')

            cog_ids = cog_df[cog_id_field].unique()

            return cog_ids

        else:
            return cog_ids


class CDR2Data:

    def __init__(self, performers, annotated):

        self.performers  = performers
        self.annotated   = annotated

    def _gcp_cdr_match_systems(self, dict_value: dict, performer: dict, annotated: bool = True) -> List:
        """
        Part of the _main function - parse through the GCP dictionary and tie the extraction to its matched system and
        its system version.

        :param dict_value: Dictionary of the GCP.
        :param performer: Dictionary of the performer information.

        :return: List containing the system, system version, and GCP dictionary value - if applicable.
        """
        sys_performer   = performer['system']
        sys_v_performer = performer['system_version']

        # If the dictionary GCP matches one of the performers and its system version
        if (dict_value['system'] == sys_performer) and (dict_value['system_version'] == sys_v_performer):
            return [sys_performer, sys_v_performer, dict_value]

        if annotated:
            sys_annotated   = self.annotated['system']
            sys_v_annotated = self.annotated['system_version']

            # If the dictionary GCP matches to the annotated system and its system version
            if (dict_value['system'] == sys_annotated) and (dict_value['system_version'] == sys_v_annotated):
                return [sys_annotated, sys_v_annotated, dict_value]

            else:
                return [None, None, None]

        # Otherwise no match or system and system version mismatch.
        else:
            return [None, None, None]

    def gcp_parse_cdr_data(self, cdr_file) -> DataFrame:

        cdr_results   = DataEng.read_json(config_file = cdr_file)

        parse_systems = (
            DataFrame(data = [self._gcp_cdr_match_systems(dict_value = c, performer = p, annotated=True)
                              for c in cdr_results for p in self.performers],
                      columns = ["system", "system_version", "data"])
            .query('system.notna() and system_version.notna() and data.notna()', engine='python')
        )

        return parse_systems


class GeorefConvert:

    def __init__(self):
        pass

    def _parse_gcps(self, cdr_dict: dict):
        """
        Parse GCP information from the CDR dictionary and construct it as a GroundControlPoint in rasterio.
        WARNING: The cdr_dict must have the exact schema; if there are changes in the CDR - then you will need to change here.

        :param cdr_dict: Dictionary of the GCPs from the CDR.

        :return: GroundControlPoint in rasterio.
        """

        return SpatialOps().construct_gcp(row = cdr_dict['rows_from_top'],
                                          col = cdr_dict['columns_from_left'],
                                          x   = cdr_dict['longitude'],
                                          y   = cdr_dict['latitude'])

    def from_gcp_data(self, parse_systems: DataFrame, system_data) -> List:
        """
        Filter the parsed GCP DataFrame based on its system data (i.e., system --> e.g., @annotated or @performer value)
        and assign GCPS by parsing it out and creating an actual GroundControlPoint via rasterio.

        :param parse_systems: Parsed GCP DataFrame.
        :param system_data: System data - performer or annotated.

        :return: List --> Filtered parsed GCP system with
        """
        gcps_df = (
            parse_systems
            .query('system == @system_data')
            .assign(gcps = lambda a: a[['data']].apply(lambda e: self._parse_gcps(*e), axis=1))
        )

        if len(gcps_df) > 0:
            # AffineTransformer from a list of GCPs
            try:
                transformer = SpatialOps().affine_transformer_from_gcps(gcps_list = gcps_df['gcps'].tolist())
            except TypeError:
                transformer = False # Set to False indicating a failed transformer

        else:
            transformer = None # Set to None as there is no data from the subset to construct GCPs.

        return [gcps_df, transformer]

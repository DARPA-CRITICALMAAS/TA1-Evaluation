"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer.

Date Created: July 2024

Date Modified: August 2024

About:

    CDR2Vec
    -------
        Parallel processing converts CDR JSON file to Vector-form (i.e., GeoDataFrame) by extracting features.
        More specifically to process CDR JSON inferenced outputs from TA1 in preparation for metric evaluation.
        WARNING: Use this only if you ran TA1 tools and want to extract outputs that have been directly saved in your
                 local directory. Otherwise, if you want to download & extract contents from the CDR use the other class
                 instead.

    FeatExtractFromCDR
    ------------------
        After downloading feature extracted items directly from the CDR, this is to convert JSON file to DataFrame.

Pre-requisites: Path and name of the CDR file. Must comply with well-defined schema standards from CriticalMAAS.
    - *_features as a property value to detect features --> applies only to CDR2Vec approach

Warnings: Does not convert pixel coordinates to geographic coordinates. Both approaches require additional processing
          for that and would need a tif file.
"""

# Data Engineering related packages
from pandas import DataFrame, concat
import numpy as np
import re

# Miscellaneous packages
from tqdm import tqdm
from typing import List, Union
from functools import partial
from multiprocessing import cpu_count, Manager

# Packages used to import custom-made packages outside of relative path.
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Custom-made packages
from ..utils import ParallelPool, DataEng, ReMethods


# Use this if the TA1 performer tools output their results directly to your output directory rather than the CDR.
class CDR2Vec:

    def __init__(self, cdr_file: str):
        """
        :param cdr_file: Name of the CDR file from TA1 inferenced outputs.
        """

        self.json_data = DataEng.read_json(config_file=cdr_file)

        # Ideally should be a nested dictionary - tested on UIUC outputs
        try:
            self.main_keys = self.json_data.keys()
            extracted      = self._mainJSONExtract()

        # If the JSON (or GeoJSON) is in a form of a list containing dictionary - proceed here; mainly for UMN/InferLink outputs (tested)
        except Exception:
            extracted = self._mainDFExtract(val = self.json_data, app_data = [])

        self.extracted = (
            extracted
            .assign(geom_size = lambda d: d[['coordinates_geometry_0']].apply(lambda e: self._geomSize(*e), axis=1))
        )

    def _mainJSONExtract(self) -> Union[DataFrame, List]:
        """
        Main CDR JSON function extraction to conversion process. Applies if the raw data comes in as a dictionary as
        opposed to list with nested dictionary.

        :return: Concatenated DataFrame of the inference features with additional information or an empty list signifying
                 there are no features to extract or schema might be incorrect.
        """
        # Iterate each dictionary key
        for k in tqdm(self.main_keys):
            val      = self.json_data[k]
            app_data = []
            # Indicate if the values extracted from the key is a list
            if isinstance(val, list):

                # If there are a list of values greater than 0, then proceed.
                if len(val) > 0:
                    return self._mainDFExtract(val = val, app_data = app_data)

    def _mainDFExtract(self, val: List, app_data: List) -> Union[DataFrame, List]:
        """
        Further extraction by converting the list with dictionary info to DataFrame.

        :param val: List containing dictionary information to be used for further extraction.
        :param app_data: An empty list that'll append final extraction conversion process.

        :return: Concatenated DataFrame of the inference features with additional information or an empty list signifying
                 there are no features to extract or schema might be incorrect.
        """
        # Convert the dictionary values to DataFrame
        tmp_df = DataFrame.from_dict(val).reset_index()

        # Looking for a column that contains "features" - could be polygon, polyline, or point *features. Schema dependent.
        tmp_col = tmp_df.columns
        coi     = [c for c in tmp_col if "_features" in c][0]

        # If there is a column containing feature, then proceed to feature extraction process.
        if coi:
            L1 = Manager().list()
            partial_func = partial(self._coiFeature, coi=coi, app_data=app_data, L1=L1)
            split_dfs = np.array_split(tmp_df, cpu_count())
            ParallelPool(start_method='spawn', partial_func=partial_func, main_list=split_dfs)

            if len(L1) > 0:
                return concat(L1)

        else:
            return []

    def _coiFeature(self, tmp_df, coi: str, app_data: List, L1):
        """
        Through the column of interest, extract geometric features and its associated properties and convert to DataFrame.
        """
        # Go through each row, acquire index value, column of interest (coi - _feature) dictionary value.
        for t in tqdm(range(len(tmp_df))):
            #get_id     = tmp_df['id'].iloc[t]
            get_idx    = tmp_df['index'].iloc[t]
            tmp_ft     = tmp_df[coi].iloc[t]

            # Convert the dictionary from the dictionary value and extract the geometry features and its associated information
            sub_df     = DataFrame.from_dict(tmp_ft)
            app_data   = self._FeatureExtract(sub_df   = sub_df,
                                              get_id   = get_idx,
                                              app_data = app_data,
                                              tmp_df   = tmp_df,
                                              coi      = coi)

        if len(app_data) > 0:
            tmp_concat = concat(app_data)
            col_df     = (
                DataFrame([list(np.where(tmp_concat.columns == c)[0]) + [c]
                           for c in tmp_concat.columns])
                .drop_duplicates(keep='first')
            )
            L1.append(self._fix_col_names(col_df=col_df, data_df=tmp_concat))

    def _FeatureExtract(self, sub_df, get_id, app_data, tmp_df, coi) -> List:
        """
        Extract the features through a series of data engineering operations.
        """
        # For each observation
        for f in range(len(sub_df)):

            # Extract the feature row, convert from dictionary to DataFrame to stack to DataFrame with transposed.
            sub_ft = sub_df['features'].iloc[f]
            fin_df = (
                DataFrame(
                    DataFrame
                    .from_dict(sub_ft)
                    .stack())
                .transpose()
            )

            # Clean up the multi-level column names by including regex.
            new_col = ["_".join(re.sub(r"[()\s+']", "", str(col)).split(","))
                       for col in fin_df.columns]

            fin_df.columns    = new_col
            fin_df['main_id'] = get_id

            # Add supplement information by merging back with the 1st level of dictionary value to DataFrame dataset.
            # Drop the column of interest field to reduce unnecessary data
            merge_df = fin_df.merge(tmp_df, left_on='main_id', right_on='index').drop(columns=[coi])
            app_data.append(merge_df)

        return app_data

    def _fix_col_names(self, col_df, data_df):
        """
        Dynamically rename column names for legibility.
        """
        new_cols = []
        for c in range(len(col_df)):
            tmp_row = list(col_df.iloc[c].dropna())
            col_iter = tmp_row[0:-1]
            col_name = tmp_row[-1]

            if len(col_iter) > 1:
                col_iter = [int(c) for c in col_iter]
                new_cols.append([col_name, col_iter])

            else:
                new_cols.append([col_name, [int(col_iter[0])]])

        fix_df = []
        for n in new_cols:
            col_name = n[0]
            iter_idx = n[1]

            sub_col = data_df[[col_name]]

            cols = []
            count = 0

            for i in iter_idx:
                cols.append(f"{col_name}_{count}")
                count += 1

            sub_col.columns = cols

            fix_df.append(sub_col)

        concat_fix = concat(fix_df, axis=1)

        return concat_fix

    def _geomSize(self, pre_geom):
        """
        Determine the size of the geometry based on the number of arrays it has.

        :param pre_geom: Pre-geometry (i.e., array containing pixel values).

        :return: Size as integer.
        """
        try:
            return np.size(pre_geom)

        except ValueError:
            return sum([np.size(pre_geom[g]) for g in range(len(pre_geom))])


# Use this if you have pulled data directly from the CDR - a bit different schema than the localized version (i.e., CDR2Vec).
class FeatExtractFromCDR:

    def __init__(self, cdr_file: str, cdr_systems: dict, feat_type: str = Union['polygon', 'point', 'line'],
                 georeferenced_data: bool = True, legend_data: bool = True):
        """
        Convert feature extracted data that was downloaded from the CDR to DataFrame. Note: It does not reproject
        pixel coordinates to geographic coordinates - that would be a separated process. Additionally, depending on
        the performer tools input, does not always guarantee to have COG ID information.

        :param cdr_file: Path to the downloaded inferenced CDR file.
        :param cdr_systems: Dictionary that must be in the following sample format below:

            cdr_systems = {
                "performers" : [{"system" : "uiuc-icy-resin", "system_version" : "0.4.6"}]
            }
        """

        performers   = cdr_systems['performers']
        tmp_data     = DataEng.read_json(config_file = cdr_file)

        # Pre-extraction process - for non-georeferenced enabled data
        if georeferenced_data is False:
            feat_df      = (
                concat([DataFrame(data = self._pre_extract_match_system(dict_value = tt, performer = p))
                       .transpose()
                       .rename(columns={0 : "system", 1 : "system_version", 2 : "data"})
                       for t in tqdm(tmp_data) for tt in t for p in performers])
                .query('system.notna() and system_version.notna() and data.notna()', engine='python')
            )

            self.pre_feat_df = feat_df

            # If there is data return as a final extracted concatenated dataframe
            if len(feat_df) > 0:
                self.extract_df = self._main_extract(feat_df = feat_df, feat_type=feat_type)

            # Otherwise as None
            else:
                self.extract_df = None

        elif georeferenced_data and legend_data:
            try:
                feat_df = (
                    concat([DataFrame(data=self._pre_extract_match_system(dict_value=t, performer=p))
                            .transpose()
                            .rename(columns={0 : "system", 1 : "system_version", 2 : "data"})
                            for t in tqdm(tmp_data) for p in performers])
                    .query('system.notna() and system_version.notna() and data.notna()', engine='python')
                )

            except TypeError:
                feat_df = (
                    concat([DataFrame(data=self._pre_extract_match_system(dict_value=tt, performer=p))
                           .transpose()
                           .rename(columns={0: "system", 1: "system_version", 2: "data"})
                            for t in tqdm(tmp_data) for tt in t for p in performers])
                    .query('system.notna() and system_version.notna() and data.notna()', engine='python')
                )

            self.pre_feat_df = feat_df

            # If there is data return as a final extracted concatenated dataframe
            if len(feat_df) > 0:
                self.extract_df = self._main_extract_georef_legend(feat_df=feat_df, feat_type=feat_type)

            # Otherwise as None
            else:
                self.extract_df = None

    def _pre_extract_match_system(self, dict_value, performer) -> List:
        """
        Pre-extraction process by extracting relevant information that matches the performer system and its version.

        :param dict_value: Current dictionary contents being extracted.
        :param performer: Performer information - system and system version.

        :return: List either containing system, system version, and dictionary contents for further extraction or None.
        """

        sys_performer = performer['system']
        sys_v         = performer['system_version']

        if (dict_value['system'] == sys_performer) and (dict_value['system_version'] == sys_v):
            return [sys_performer, sys_v, dict_value]
        else:
            return [None, None, None]

    def _main_extract(self, feat_df: DataFrame, feat_type) -> DataFrame:
        """
        Main extraction function to further extract data items.

        :param feat_df: DataFrame that has been pre-extracted.

        :return: Concatenated DataFrame in the following schema:
            - system            => Performer system
            - system_version    => System version of the performer
            - cog_id            => COG ID
            - px_bbox           => Bounding Box of the Legend Item --> highly critical to determine which geologic label it is.
            - type              => Geometry type instance, such as Polygon, LineString, or Point
            - coordinates       => Pixel coordinates associated to the type
        """

        if feat_type == "polygon" or feat_type == "line":

            order_cols  = ['system', 'system_version', 'cog_id', 'p_num_geom', 'type', 'coordinates', 'abbreviation', 'color',
                           'pattern', 'px_bbox']
            append_data = []
            # Iterate through each row of the DataFrame
            for f in tqdm(range(len(feat_df))):
                tmp_row     = feat_df.iloc[f]

                # Important system information
                system      = tmp_row['system']
                sys_v       = tmp_row['system_version']

                # Data contents to be extracted and converted to DataFrame
                tmp_data    = tmp_row['data']
                #pre_df      = DataFrame({'cog_id' : [tmp_data['cog_id']], 'px_bbox' : [tmp_data['px_bbox']]})
                pre_df      = DataFrame({'cog_id' : [tmp_data['cog_id']]})

                if feat_type == "line":

                    px_geojson = (
                        DataFrame
                        .from_dict(tmp_data['px_geojson'])
                        .groupby(['type'], as_index=False)
                        .agg({"coordinates" : lambda x: x.tolist()})
                    )

                    num_geom = len(px_geojson)

                else:
                    px_geojson = DataFrame.from_dict(tmp_data['px_geojson'])
                    num_geom   = len(px_geojson)

                legend_data = tmp_data['legend_item']
                legend_df = DataFrame({'abbreviation' : [legend_data['abbreviation']],
                                       'color'        : [legend_data['color']],
                                       'pattern'      : [legend_data['pattern']],
                                       'px_bbox'      : [legend_data['px_bbox']]})


                # Concatenate both DataFrames and add system information to it; re-arrange columns for organizational purposes.
                fin_df = (
                    concat([pre_df.reset_index(), px_geojson.reset_index(), legend_df.reset_index()], axis=1)
                    .assign(system=system, system_version=sys_v, p_num_geom=num_geom)
                    [order_cols]
                )

                # Append concatenated DataFrame to list
                append_data.append(fin_df)

            # Concat all list of DataFrames into one.
            concat_df = concat(append_data)

            return concat_df

        elif feat_type == "point":

            order_cols = ['system', 'system_version', 'cog_id', 'px_bbox', 'type', 'coordinates']
            append_data = []
            # Iterate through each row of the DataFrame
            for f in tqdm(range(len(feat_df))):
                tmp_row = feat_df.iloc[f]

                # Important system information
                system = tmp_row['system']
                sys_v = tmp_row['system_version']

                # Data contents to be extracted and converted to DataFrame
                tmp_data = tmp_row['data']
                pre_df = DataFrame({'cog_id': [tmp_data['cog_id']], 'px_bbox': [tmp_data['px_bbox']]})
                px_geojson = DataFrame.from_dict(tmp_data['px_geojson']).groupby(['type'], as_index=False).agg({'coordinates' : lambda x: x.tolist()})

                # Concatenate both DataFrames and add system information to it; re-arrange columns for organizational purposes.
                fin_df = (
                    concat([pre_df.reset_index(), px_geojson.reset_index()], axis=1)
                    .assign(system=system, system_version=sys_v, p_num_geom=1)
                    [order_cols]
                )

                # Append concatenated DataFrame to list
                append_data.append(fin_df)

            # Concat all list of DataFrames into one.
            concat_df = concat(append_data)

            return concat_df

    def _main_extract_georef_legend(self, feat_df: DataFrame, feat_type):
        order_cols = ['system', 'system_version', 'cog_id',
                      'p_num_geom', 'type', 'coordinates',
                      'abbreviation', 'color', 'pattern', 'px_bbox']

        append_data = []
        # Iterate through each row of the DataFrame
        for f in tqdm(feat_df.itertuples(index=True), total=feat_df.shape[0]):

            # Important system information
            system   = f.system
            sys_v    = f.system_version
            tmp_data = f.data

            # Pre-extraction
            cog_id      = tmp_data['cog_id']
            legend_item = tmp_data['legend_item']
            abbrev      = legend_item['abbreviation']
            pattern     = legend_item['pattern']
            color       = legend_item['color']
            px_bbox     = legend_item['px_bbox']

            if len(abbrev) == 0:
                abbrev = " ".join(re.sub(f"{cog_id}_{system}_{sys_v}_", "", tmp_data['legend_id']).split("_")[1:])

            pre_df = DataFrame([[cog_id, abbrev, pattern, color, px_bbox]], columns=['cog_id', 'abbreviation',
                                                                                     'pattern', 'color', 'px_bbox'])

            if feat_type == "point":
                if len(tmp_data['projected_feature']) > 0:
                    geojson = concat([DataFrame({'type'        : [p['projected_geojson']['type']],
                                                 'coordinates' : [p['projected_geojson']['coordinates']],
                                                 'p_num_geom'  : [1]})
                                      for p in tmp_data['projected_feature']])

                else:
                    geojson = DataFrame({'type'        : [feat_type],
                                         'coordinates' : [None],
                                         'p_num_geom'  : [0]})

            elif feat_type == "line" or feat_type == "polygon":

                if len(tmp_data['projected_feature']) > 0:
                    coordinates = DataFrame([[pp, len(pp), e]
                                             for e,p in enumerate(tmp_data['projected_feature'])
                                             for pp in p['projected_geojson']['coordinates']], columns=['coordinates',
                                                                                                        'p_num_geom',
                                                                                                        'index'])

                    geojson = concat([DataFrame({'type' : [p['projected_geojson']['type']],
                                                 'index' : [e]})
                                      for e,p in enumerate(tmp_data['projected_feature'])])

                    geojson = coordinates.merge(geojson, on=['index'])

                    if feat_type == "line":
                        geojson = (
                            geojson
                            .groupby(['type', 'index', 'p_num_geom'], as_index=False)
                            .agg({'coordinates' : lambda x: x.tolist()})
                        )

                else:
                    geojson = DataFrame({'type'        : [feat_type],
                                         'index'       : [None],
                                         'coordinates' : [None],
                                         'p_num_geom'  : [0]})

            fin_df = (
                concat([pre_df.reset_index(), geojson.reset_index()], axis=1)
                .assign(system=system, system_version=sys_v)
                [order_cols]
            )

            append_data.append(fin_df)

        concat_df = concat(append_data)

        return concat_df


# Use this to extract Legend annotated items that was downloaded from the CDR.
class LegendAnnotatedExtract:

    def __init__(self, cdr_file: str, key_interest_field: List = ["cog_id", "system", "system_version", "category",
                                                                  "label", "abbreviation", "description", "pattern",
                                                                  "color", "px_bbox"]):
        """
        Extract Legend annotated items that was downloaded from the CDR.

        :param cdr_file: JSON file downloaded from the CDR.
        :param key_interest_field: List of keys that are interested for extraction (i.e, to obtain). Default is provided.
                                   If need to change, acquire inspecting the legend annotated schema.
        """
        # Read Legend annotated
        json_data = DataEng.read_json(config_file = cdr_file)

        # Iterate through key interests and retain those - convert to DataFrame
        legend_df = (
            concat([DataFrame({l : [e[l]], 'index' : [i]})
                    for i,e in enumerate(json_data)
                    for l in key_interest_field
                    if l in e.keys()])
            .groupby('index', as_index=False)
            .apply(lambda a: a.stack().transpose())
        )

        # Clean up the DataFrame and remove unwanted byproduct columns
        legend_df.columns = legend_df.columns.droplevel(0)
        list_cols         = list(legend_df.columns)
        idx_cols          = [i for i,e in enumerate(list_cols) if "index" in e]
        self.legend_df    = legend_df.drop(legend_df.columns[idx_cols], axis=1).drop(columns=[''])

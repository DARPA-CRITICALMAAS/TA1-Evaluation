"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer.

Date Created: September 2024.

Last Update: September 2024.

About:
    - EvalSet
    - EvalUpdates
"""

from typing import Union, List
from .utils import DataEng, ParallelPool
from .ground_truth_inventory import EvalChecks
from pandas import read_csv, DataFrame, concat
import numpy as np
from multiprocessing import cpu_count, Manager
from functools import partial
from tqdm import tqdm
import shutil
import warnings


class EvalSet:

    def __init__(self, kwargs: Union[str, dict]):
        if isinstance(kwargs, str):
            kwargs = DataEng.read_json(config_file = kwargs)

        self.georef_path  = kwargs['georef_path']
        self.feature_path = kwargs['feature_path']

        # Current Evaluation Master Set
        self.eval_df        = read_csv(kwargs['eval_csv'])
        self.id_field       = kwargs['id_field']
        self.cog_field      = kwargs['cog_field']
        self.source_field   = kwargs['source_field']
        self.georef_field   = kwargs['georef_field']
        self.ft_extract_fld = kwargs['ft_extract_field']

        """
        Check if there is an inventory file
        """

        """
        Georeferencing Evaluation set 
            - georef => the entire georeferencing set 
            - g_topo => List of topo maps 
            - g_maps => List of NGMDB maps
        """
        self.georef = self.eval_df[self.eval_df[self.georef_field] == "Yes"]
        self.g_topo = self.georef[self.georef[self.source_field] == "Topo"]
        self.g_maps = self.georef[self.georef[self.source_field] == "Map"]

        """
        Georeferencing - Pre-inventory & what's missing. 
            - Most important is: pre_inventory_georef.
        """
        self.prep_invent_gtif = self._prep_inventory(path     = self.georef_path,
                                                     file_ext = ".tif",
                                                     id_list  = self.georef[self.id_field].to_list(),
                                                     df       = self.georef)

        self.pre_inventory_georef = self._pre_inventory(eval_type = "Georeferencing")

        """
        Feature Extraction Evaluation Set
            - feat_ext => the entire feature extraction set 
        """
        self.feat_ext = self.eval_df[(self.eval_df[self.ft_extract_fld] == "Yes") & (self.eval_df[self.source_field] == "Map")]

        """
        Feature Extraction - Pre-inventory & what's missing for binary rasters (ftif), shapefiles (fshp), and feather (ffeather).
            - Most important is pre_inventory_ftext. 
        """
        self.prep_invent_ftif = self._prep_inventory(path     = self.feature_path,
                                                     file_ext = ".tif",
                                                     id_list  = self.feat_ext[self.id_field].to_list(),
                                                     df       = self.feat_ext)

        self.prep_invent_fshp = self._prep_inventory(path     = self.feature_path,
                                                     file_ext = ".shp",
                                                     id_list  = self.feat_ext[self.id_field].to_list(),
                                                     df       = self.feat_ext)

        self.prep_invent_ffeather = self._prep_inventory(path     = self.feature_path,
                                                         file_ext = ".feather",
                                                         id_list  = self.feat_ext[self.id_field].to_list(),
                                                         df       = self.feat_ext)

        self.pre_inventory_ftext = self._pre_inventory(eval_type = "Feature Extraction")

    def _prep_inventory(self, path: str, file_ext: str, id_list: List, df: DataFrame) -> List:
        """
        Prepare inventory of missing items per evaluation type set.

        :param path: Path to where the files are stored for evaluation.
        :param file_ext: File extension.
        :param id_list: List of IDs from the evaluation type.
        :param df: DataFrame from the subset of the evaluation set.

        :return: List of information.
            - curr_invent => Current inventory that checks extension files from designated path and compares with the
                             evaluation type set (i.e., Georeferencing or Feature Extraction).
            - missing_ids => IDs that exist, but are missing the file extension.
            - diff_ids    => IDs that do not exist in the designated eval. path and are actively missing.
            - final_ids   => Combination of missing_ids and diff_ids.
        """

        """
        Current inventory by checking designated path that contain extension files and
        compare with the evaluation set, specifically those catered for a process type.
        """
        curr_invent = EvalChecks().regex_update_match(path     = path,
                                                      id_list  = id_list,
                                                      file_ext = file_ext)

        ids_have_file  = curr_invent.query('has_file == True')['id'].tolist()
        ids_pot_nofile = curr_invent.query('has_file == False')['id'].tolist()

        # IDs that exist, but are missing the file extension.
        missing_ids = EvalChecks().check_missing(current_set = ids_have_file,
                                                 new_set     = ids_pot_nofile)

        # diff_ids => IDs that do not exist in the designated path and are actively missing
        diff_ids    = list(set(list(df[self.id_field])) - set(list(curr_invent['id'])))
        final_ids   = diff_ids + missing_ids

        return [curr_invent, missing_ids, diff_ids, final_ids]

    def _pre_inventory(self, eval_type: str = "Feature Extraction" or "Georeferencing") -> DataFrame:
        """
        Pre-inventory (before an update of new data occurs) for the evaluation type set.

        :param eval_type: Evaluation type - only two values ("Feature Extraction" or "Georeferencing").

        :return: DataFrame of the inventory updates.
        """

        concat_info = []

        if eval_type == "Feature Extraction":
            # List of feature extraction IDs; followed by a dictionary containing the ID (as key) and source (as value).
            feat_ids  = self.feat_ext[self.id_field].tolist()
            feat_dict = {f : s for f,s in zip(self.feat_ext[self.id_field], self.feat_ext[self.source_field])}

            # List of IDs exists in directory but missing TIF files; same applies for shp files.
            missing_tif_in_dir = self.prep_invent_ftif[1]
            missing_shp_in_dir = self.prep_invent_fshp[1]

            # List of IDs that do not exist in the directory and are missing TIF files; same applies for shp.
            missing_tif     = self.prep_invent_ftif[2]
            missing_shp     = self.prep_invent_fshp[2]

            # List of IDs that overall are missing feather files for value matching inventory to the respective shapefile.
            # Applies for both - IDs that exist in directory and IDs that do not exist but are missing.
            missing_feather = self.prep_invent_ffeather[3]

            # Run through all feature extraction IDs and populate current updates.
            for f in feat_ids:
                tmp_dict = {f : {"in_dir_t"     : True,
                                 "in_dir_s"     : True,
                                 "has_tif"      : True,
                                 "has_shp"      : True,
                                 "value_match"  : True,
                                 "process_type" : "Feature Extraction",
                                 "source"       : None}}

                if f in missing_tif_in_dir:
                    tmp_dict[f]['has_tif'] = False

                elif f in missing_tif:
                    tmp_dict[f]['in_dir_t'] = False
                    tmp_dict[f]['has_tif']  = False

                if f in missing_shp_in_dir:
                    tmp_dict[f]['has_shp'] = False

                elif f in missing_shp:
                    tmp_dict[f]['in_dir_s'] = False
                    tmp_dict[f]['has_shp']  = False

                if f in missing_feather:
                    tmp_dict[f]['value_match'] = False

                tmp_dict[f]['source'] = feat_dict[f]
                concat_info.append(DataFrame(tmp_dict).transpose())

            concat_df = concat(concat_info).reset_index().rename(columns={'index': 'id'})

            return concat_df

        elif eval_type == "Georeferencing":
            # List of georeferencing IDs; followed by a dictionary containing the ID (as key) and source (as value).
            georef_ids  = self.georef[self.id_field].tolist()
            georef_dict = {g : s for g,s in zip(self.georef[self.id_field], self.georef[self.source_field])}

            # List of IDs exists in directory but missing TIF files; followed by list of IDs that do not exist in the
            # directory and are missing TIF files
            missing_tif_in_dir = self.prep_invent_gtif[1]
            missing_tif        = self.prep_invent_gtif[2]

            # Run through all georeferencing IDs and populate current updates.
            for g in georef_ids:
                tmp_dict = {g : {"in_dir"       : True,
                                 "has_tif"      : True,
                                 "process_type" : "Georeferencing",
                                 "source"       : None}}

                if g in missing_tif_in_dir:
                    tmp_dict[g]['has_tif'] = False

                elif g in missing_tif:
                    tmp_dict[g]['in_dir']  = False
                    tmp_dict[g]['has_tif'] = False

                tmp_dict[g]['source'] = georef_dict[g]
                concat_info.append(DataFrame(tmp_dict).transpose())

            concat_df = concat(concat_info).reset_index().rename(columns = {'index' : 'id'})

            return concat_df


class EvalUpdates(EvalSet):

    def __init__(self, kwargs: Union[str, dict]):

        if isinstance(kwargs, str):
            kwargs = DataEng.read_json(config_file = kwargs)

        # Executing a child class - pre-inventory of what is currently available and what is missing.
        super().__init__(kwargs)

        self.updated_data       = kwargs.get('recent_data', None)
        self.binary_vector_info = self._match_binary_shp()

        #self.value_match = EvalChecks().check_file_ext(path     = self.feature_path,
        #                                               file_ext = ".feather")

        if self.updated_data is not None:
            """
            Check updates for tif & shp files in feature extraction 
            """
            self.tif_update_info_ft = self._check_updates(id_list  = self.prep_invent_ftif[3],
                                                          file_ext = ".tif")

            if self.tif_update_info_ft is not None:
                if len(self.tif_update_info_ft[1]) > 0:
                    tmp_ids = self.tif_update_info_ft[1]['id'].unique()
                    self._update_data(inventory = self.pre_inventory_ftext,
                                      new_ids   = tmp_ids,
                                      cols      = ['in_dir_t', 'has_tif'])

                    # Conduct movement
                    self._move_data(update_df = self.tif_update_info_ft[1],
                                    eval_path = self.feature_path,
                                    new_ids   = tmp_ids)

            self.shp_update_info_ft = self._check_updates(id_list  = self.prep_invent_fshp[3],
                                                          file_ext = ".shp")

            if self.shp_update_info_ft is not None:
                if len(self.shp_update_info_ft[1]) > 0:
                    tmp_ids = self.shp_update_info_ft[1]['id'].unique()
                    self._update_data(inventory = self.pre_inventory_ftext,
                                      new_ids   = tmp_ids,
                                      cols      = ['in_dir_s', 'has_shp'])

                    # Conduct movement
                    self._move_data(update_df = self.shp_update_info_ft[1],
                                    eval_path = self.feature_path,
                                    new_ids   = tmp_ids)

                    # Conduct binary raster - vector matching - Maybe re-run tool again

            """
            Check updates for tif files in georeferencing
            """
            self.tif_update_info_georef = self._check_updates(id_list  = self.prep_invent_gtif[3],
                                                              file_ext = ".tif")

            if self.tif_update_info_georef is not None:
                if len(self.tif_update_info_georef[1]) > 0:
                    tmp_ids = self.tif_update_info_georef[1]['id'].unique()
                    self._update_data(inventory = self.pre_inventory_georef,
                                      new_ids   = tmp_ids,
                                      cols      = ['in_dir', 'has_tif'])

                    # Conduct movement
                    self._move_data(update_df = self.tif_update_info_georef[1],
                                    eval_path = self.georef_path,
                                    new_ids   = tmp_ids)

    def _clean_geog_L3(self, L3):
        fin_concat = []
        for o in range(len(L3)):
            try:
                fin_concat.append(concat([concat(t[0]).assign(id=t[1]) for t in L3[o]]).reset_index().drop(columns=['index']))
            except ValueError:
                pass

        if len(fin_concat) > 0:
            fin_concat = concat(fin_concat).reset_index().drop(columns=['index'])
            return fin_concat

        else:
            return None

    def _match_binary_shp(self) -> List:
        """
        Parallel process to match geologic values obtained from the binary rasters to the vector values from the shapefile.

        :return: List (if there are files that need to be processed).
        """
        missing_value_ids = (
            self.pre_inventory_ftext
            .query('has_tif == True and has_shp == True and value_match == False')
            ['id']
            .tolist()
        )

        if len(missing_value_ids) > 0:

            split_ids    = np.array_split(missing_value_ids, cpu_count())
            L1           = Manager().list()
            L2           = Manager().list()
            L3           = Manager().list()
            partial_func = partial(self._parallel_match, L1 = L1, L2 = L2, L3 = L3)
            ParallelPool(start_method="spawn", partial_func=partial_func, main_list=split_ids, num_cores=cpu_count())
            concat_L1    = concat(L1).reset_index().drop(columns=['index'])
            concat_L2    = concat(L2).reset_index().drop(columns=['index'])

            if L3 is not None:
                L3 = self._clean_geog_L3(L3 = L3)

            return [missing_value_ids, concat_L1, concat_L2, L3]

        else:
            return [None, None, None, None]

    def _parallel_match(self, id_list, L1, L2, L3):
        """
        Parallel process function to match geologic values obtained from binary rasters to vector values.

        :param id_list: List of IDs to be processed.
        :param L1: List manager that appends unmatched binary geologic values. Ones that have not succeeded.
        :param L2: List manager that appends matched binary geologic values to their corresponding vector value, which
                   field in the shapefile, and which shapefile is it per ID.
        """

        warnings.filterwarnings(action='ignore', category=FutureWarning)

        add_info, concat_info, tmp_data, other_geo_values = [],[],[],[]
        for i in id_list:
            prep_data = EvalChecks().binary_geologic_data(path = self.feature_path,
                                                          ngmdb_id = i)

            shp_data    = prep_data[0]
            binary_data = prep_data[1]

            # Preparation for the binary geologic value to the vector value process.
            # Setting a nested list - one to lower to increase similarity matching.
            unique_geologic     = [[b.lower(), b] for b in binary_data['clean_geologic'].unique()]
            use_unique_geologic = [b[1] for b in unique_geologic]

            # Match binary geologic value to the vector value process - setting threshold to 0.9.
            fin_match_info      = EvalChecks().match_geologic_value_to_field_shp(shp_files       = shp_data['path'],
                                                                                 unique_geologic = unique_geologic,
                                                                                 threshold       = 0.9)

            # Add successes to the DataFrame, identify which values have not succeeded in matching
            df = (
                DataFrame(fin_match_info[0], columns = ['binary_value', 'field_name', 'geo_value_vector',
                                                        'matched_ratio', 'shp_file', 'match_method', 'geom_type',
                                                        'read_corrupt'])
                .assign(id = i)
            )
            missing_set  = list(set(use_unique_geologic) - set(df['binary_value']))
            perc_success = round((len(use_unique_geologic) - len(missing_set)) / len(use_unique_geologic), 3)
            #value_file   = f"{self.feature_path}/{i}/value_field_match.feather"

            # Compress the matched values in a list - grouping by shapefile and field name.
            order_cols = ['id', 'binary_value', 'geo_value_vector', 'field_name', 'matched_ratio', 'shp_file',
                          'match_method', 'geom_type', 'read_corrupt']
            grp_df     = (
                df
                .drop(columns=['id'])
                .groupby(['shp_file', 'field_name', 'match_method', 'geom_type', 'read_corrupt'], as_index=False)
                .agg({'binary_value'      : lambda x: x.tolist(),
                      'geo_value_vector'  : lambda x: x.tolist(),
                      'matched_ratio'     : lambda x: x.tolist()})
                .assign(id = i)
                [order_cols]
            )

            #other_geog_values = DataFrame(fin_match_info[1], columns=['binary_value', 'cand_values', 'field', 'shp_file']).assign(id = i)

            add_info.append([i, perc_success, missing_set])
            concat_info.append(grp_df)
            tmp_data.append(df)

            if fin_match_info[1] is not None:
                other_geo_values.append([fin_match_info[1], i])

        # Append the information
        concat_pre_data  = concat(tmp_data)
        matched_binary   = concat(concat_info)
        unmatch_binary   = (
            DataFrame(add_info, columns = ['id', 'success_rate', 'unmatched_binary'])
            .merge(concat_pre_data, on=['id'], how='left')
            .explode('unmatched_binary')
            [['id', 'success_rate', 'unmatched_binary', 'read_corrupt']]
            .drop_duplicates(['id', 'success_rate', 'unmatched_binary', 'read_corrupt'])
            .reset_index()
            .drop(columns=['index'])
            .groupby(['id', 'success_rate', 'read_corrupt'], as_index=False)
            .agg({'unmatched_binary' : lambda x: x.tolist()})
        )

        del tmp_data

        L1.append(unmatch_binary)
        L2.append(matched_binary)

        if other_geo_values is not None:
            L3.append(other_geo_values)

    def _check_updates(self, id_list, file_ext):
        try:
            new_feat = EvalChecks().regex_update_match(path     = self.updated_data,
                                                       id_list  = id_list,
                                                       file_ext = file_ext)

            if new_feat is not None:
                if len(new_feat) > 0:
                    # In preparation to move their directory and update inventory
                    sub_files = new_feat.query('has_file == True')

                    missing_feat = EvalChecks().check_missing(current_set = id_list,
                                                              new_set     = list(new_feat['id']))

                    tif_update_feat_unmatch = EvalChecks().unmatch(current_set = id_list,
                                                                   new_set     = list(new_feat['id']))

                    return [new_feat, sub_files, missing_feat, tif_update_feat_unmatch]

                else:
                    return None

            else:
                return None

        except ValueError:
            return None

    def _update_data(self, inventory, new_ids, cols):
        for n in new_ids:
            for col in cols:
                inventory.loc[inventory['id'] == n, col] = True

    def _move_data(self, update_df, eval_path, new_ids):
        for t in tqdm(new_ids):
            sub_df   = update_df.query('id == @t').drop_duplicates('directory')
            dir_name = f"{eval_path}/{t}"
            DataEng.checkdir(dir_name = dir_name)

            for d in sub_df['directory']:
                shutil.move(d, dir_name)
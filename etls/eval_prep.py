"""

"""

from .utils import DataEng, ParallelPool
from .eval_inventory import EvalUpdates
from .ground_truth_inventory import EvalChecks
from pandas import DataFrame, concat
from geopandas import read_file
from typing import Union, List
from functools import partial
from tqdm import tqdm
from multiprocessing import Manager, cpu_count
from numpy import array_split


class PrepEval:

    def __init__(self, kwargs: Union[str, List], output_path: str):

        if isinstance(kwargs, str):
            kwargs = DataEng.read_json(config_file = kwargs)

        updated_data        = kwargs.get('recent_data', None)

        """
        Inventory Process for the Ground Truth set (Feature Extraction & Georeferencing) 
        """
        self.inventory_data = self._inventory_run(updated_data = updated_data,
                                                  kwargs       = kwargs)

        pre_eval_dict = self.inventory_data.eval_df[[kwargs['id_field'], kwargs['cog_field']]].drop_duplicates([kwargs['id_field'],
                                                                                                                kwargs['cog_field']])

        self.eval_dict = {i : c for i,c in zip(pre_eval_dict[kwargs['id_field']], pre_eval_dict[kwargs['cog_field']])}

        ft_ext_inventory        = self.inventory_data.pre_inventory_ftext
        self.in_house_ft_invent = ft_ext_inventory.query('has_tif == True and has_shp == True')
        self.missing_ft_invent  = ft_ext_inventory.query('has_tif == False or has_shp == False')

        georef_inventory            = self.inventory_data.pre_inventory_georef
        self.in_house_georef_invent = georef_inventory.query('has_tif == True')
        self.missing_georef_invent  = georef_inventory.query('has_tif == False')

        """
        Binary-Vector value matching and using that to compile as geologic values list. 
        This is intended to be used as guidance for feature extraction pre-processing and evaluation. 
        """
        self.match_binary_vector    = self.inventory_data.binary_vector_info[2]
        self.unmatch_binary_vector  = self.inventory_data.binary_vector_info[1]
        self.other_geo_value_per_id = self.inventory_data.binary_vector_info[3]
        self.compiled_geologic_list = self._geologic_value_list()

        self.lne_pnt_ontolog             = EvalChecks().lne_pnt_ontolog
        crude_match_process              = self._build_fallback_mechanism()
        self.crude_match                 = crude_match_process[0].rename(columns={'path' : 'shp_file'})
        self.unmatch_binary_vector_final = crude_match_process[1]

        # Export important data for later use and viewing - will need to do the same for georeferencing
        self.in_house_ft_invent.to_parquet(f"{output_path}/ft_inhouse_inventory.parquet")
        self.match_binary_vector.to_parquet(f"{output_path}/ft_match_binary.parquet")
        self.crude_match.to_parquet(f"{output_path}/ft_crude_match.parquet")
        self.unmatch_binary_vector_final.to_parquet(f"{output_path}/ft_unmatch_binary.parquet")
        self.inventory_data.prep_invent_ftif[0].to_parquet(f"{output_path}/ft_tif_files.parquet")
        DataEng.write_json(output_file=f"{output_path}/cog_id_info.json", data=self.eval_dict)

    def _inventory_run(self, updated_data, kwargs):
        # Potentially run twice the evaluation set inventory & updates.
        # Once - to create pre-inventory & move any new updates; if there
        # are updates, re-run to finalize the inventory.
        if updated_data is not None:
            for rerun in range(0,2):
                data = EvalUpdates(kwargs = kwargs)

        else:
            data = EvalUpdates(kwargs = kwargs)

        return data

    def _geologic_value_list(self):

        unmatched_data = self.unmatch_binary_vector.explode('unmatched_binary')
        matched_data   = (
            self.match_binary_vector
            .explode(['binary_value', 'geo_value_vector', 'matched_ratio'])
            .drop_duplicates(['binary_value', 'geo_value_vector', 'matched_ratio'])
            .sort_values('binary_value')
        )

        binary_values = matched_data['binary_value'].tolist() + unmatched_data['unmatched_binary'].tolist()
        vector_values = matched_data['geo_value_vector'].tolist() + [None] * len(unmatched_data)
        matched_ratio = matched_data['matched_ratio'].tolist() + [None] * len(unmatched_data)

        compiled_geologic = (
            DataFrame({'binary_values'    : binary_values,
                       'geo_value_vector' : vector_values,
                       'matched_ratio'    : matched_ratio})
            .assign(geometry_type = None)
        )

        return compiled_geologic

    def _crude_match(self, label):
        tmp_append = []
        try:
            for f in self.lne_pnt_ontolog[label]:
                tmp_append.append(f['feature_type'])

            return tmp_append

        except KeyError:
            return "Not in Dictionary"

    def _match_by_geom(self, path, geom_type):
        data_info = []
        for p in path:
            shp = read_file(p)
            geom = shp.geom_type[0]
            if geom == geom_type:
                data_info.append(p)


        if len(data_info) > 0:
            return data_info

        else:
            return "Does Not Exist In Ground-Truth"

    def _build_fallback_mechanism(self):
        match_binary_vec   = self.match_binary_vector[['id', 'shp_file', 'geom_type']]
        geom_type          = lambda x: self._crude_match(label=x)
        corrupt_data       = (
            self.unmatch_binary_vector
            .query('read_corrupt != "no_corrupt"')
            .explode('unmatched_binary')
            .assign(geom_type = "Unknown",
                    reason    = 'corrupt ground-truth shapefile')
            [['id', 'unmatched_binary', 'geom_type', 'reason']]
        )

        unmatch_binary_vec = (
            self.unmatch_binary_vector
            .query('read_corrupt == "no_corrupt"')
            [['id', 'unmatched_binary']]
            .explode('unmatched_binary')
            .assign(geom_type = lambda d: list(map(geom_type, d['unmatched_binary'])))
        )

        in_dict = (
            unmatch_binary_vec
            .query('geom_type != "Not in Dictionary"')
            .explode('geom_type')
        )

        not_in_dict = (
            unmatch_binary_vec
            .query('geom_type == "Not in Dictionary"')
            .assign(geom_type = "Unknown",
                    reason = "Not in Dictionary")
            [['id', 'unmatched_binary', 'geom_type', 'reason']]
        )

        unmatch_ids   = in_dict['id'].unique()
        shp_file_list = (
            self.inventory_data.prep_invent_fshp[0]
            .query('has_file == True and id in @unmatch_ids')
            [['id', 'path', 'filename']]
        )

        pre_crude_data = in_dict.merge(shp_file_list, on=['id'])
        pre_crude_data = array_split(pre_crude_data, cpu_count())
        L1             = Manager().list()
        L2             = Manager().list()
        partial_func   = partial(self._parallel_fallback, L1 = L1, L2 = L2)
        ParallelPool(start_method='spawn', partial_func=partial_func, main_list=pre_crude_data)
        crude_match = concat(L1)
        no_match    = concat([concat(L2), not_in_dict, corrupt_data]).reset_index().drop(columns=['index'])

        return [crude_match, no_match]

    def _parallel_fallback(self, pre_crude_data: DataFrame, L1, L2):

        tqdm.pandas()

        crude_match = (
            pre_crude_data
            .groupby(['id', 'unmatched_binary', 'geom_type'], as_index=False)
            .progress_apply(lambda a: concat([DataFrame({'id'               : [a['id'].iloc[0]],
                                                         'unmatched_binary' : [a['unmatched_binary'].iloc[0]],
                                                         'geom_type'        : [a['geom_type'].iloc[0]],
                                                         'path'             : [self._match_by_geom(path=a['path'],
                                                                                                   geom_type=a['geom_type'].iloc[0])]})]))
            .explode('path')
        )

        no_cand = (
            crude_match
            .query('path == "Does Not Exist In Ground-Truth"')
            .rename(columns={'path' : 'reason'})
            [['id', 'unmatched_binary', 'geom_type', 'reason']]
        )

        crude_match = crude_match.query('path != "Does Not Exist In Ground-Truth"')

        L1.append(crude_match)
        L2.append(no_cand)

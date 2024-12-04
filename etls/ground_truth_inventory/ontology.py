"""

"""

from pandas import read_csv, DataFrame, concat
from typing import Union
from tqdm import tqdm

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ..utils import DataEng, ReMethods
from ..ground_truth_inventory import EvalChecks


class OntoGeog:

    def __init__(self, kwargs: Union[str, dict]):

        if isinstance(kwargs, str):
            kwargs = DataEng.read_json(config_file=kwargs)

        self.onto_file         = (
            read_csv(kwargs['onto_file'])
            .assign(alt_name=lambda d: d['alternate_example'].str.split(","))
            .explode('alt_name')
            .assign(alt_name=lambda d: d['alt_name'].str.lstrip().str.rstrip())
            [['label', 'ftr_type', 'geometry_type', 'alt_name']]
        )
        self.ground_truth_path = kwargs['ground_truth_set']
        self.binary_rasters    = self._binary_rasters()

        # NOTE: THIS HAD TO BE DONE MANUALLY TO CREATE THE 1st VERSION OF THIS SET
        self.update_set = kwargs.get('update_set', None)

        if self.update_set is None:
            print('Creating file that does not exist and requires manual input to determine what geologic feature is \n'
                  'a polygon type. After completion, save & exit the excel file and re-run this program. The file has \n'
                  'been exported to ' + f"{self.ground_truth_path}/update_geologic_values.csv")

            tmp_df = DataFrame({'binary_raster' : self.binary_rasters['geologic'].unique()}).assign(feat_type=None)
            tmp_df.to_csv(f"{self.ground_truth_path}/update_geologic_values.csv")
            sys.exit()

        else:
            self.update_geologic_set = read_csv(self.update_set).query('feat_type != "poly"')
            bin_unique_geolog        = self.binary_rasters['geologic'].unique()
            self.bin_df              = DataFrame(bin_unique_geolog, columns=['binary_raster'])
            self.onto_df_match       = self._dataCand(bin_unique_geolog=bin_unique_geolog)
            # Need Gabrielle to cross-check if these are correct
            self.missing_onto_geom   = self._identify_missing()

    def _binary_rasters(self):

        binary_rasters = (
            EvalChecks()
            .check_file_types(path = self.ground_truth_path, file_ext=".tif")
            .query('has_file == True')
            .assign(geologic  = lambda b: b['filename'].str.split("_").apply(lambda e: e[1:-1][0]),
                    feat_type = lambda b: b['filename'].str.split("_").apply(lambda e: e[-1].split(".tif")[0]))
            [['id', 'path', 'directory', 'filename', 'geologic', 'feat_type']]
        )

        return binary_rasters

    def _accept_val(self, value, threshold):
        if value > threshold:
            return value

        else:
            return None

    def _dataCand(self, bin_unique_geolog):

        ont_label_set = [str(l) for l in self.onto_file['label'].unique()]
        ont_ftr_type  = [str(o) for o in self.onto_file['ftr_type'].unique()]
        ont_alt_name  = [str(s) for s in self.onto_file['alt_name'].unique()]

        data_cand     = []
        for b in tqdm(bin_unique_geolog):

            g_l = ReMethods().max_sequence_matcher(str1=b.lower(), str2_list=ont_label_set)
            g_f = ReMethods().max_sequence_matcher(str1=b.lower(), str2_list=ont_ftr_type)
            g_a = ReMethods().max_sequence_matcher(str1=b.lower(), str2_list=ont_alt_name)

            tmp_cand = [[b, "label"] + g_l, [b, "ftr_type"] + g_f, [b, "alt_name"] + g_a]

            try:
                list_cand = concat([DataFrame([t], columns=['geo_value', 'field', 'match', 'threshold'])
                                    for t in tmp_cand
                                    if self._accept_val(value=t[3], threshold=0.8) is not None])

                if len(list_cand) > 0:
                    data_cand.append(list_cand)

            except ValueError:
                pass

        onto_df_match = concat(data_cand)
        est_geom      = []
        for o in range(len(onto_df_match)):
            tmp_row = onto_df_match.iloc[o]
            field = tmp_row['field']
            match = tmp_row['match']
            feat_type = self.onto_file[self.onto_file[field] == match]['geometry_type'].iloc[0]
            est_geom.append(feat_type)

        onto_df_match['geometry_type'] = est_geom

        return onto_df_match

    def _identify_missing(self):
        address_set = list(set(self.update_geologic_set['binary_raster']) - set(self.onto_df_match['geo_value']))
        address_df  = self.update_geologic_set.query('binary_raster in @address_set')
        return address_df
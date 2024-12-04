from typing import List, Union
from functools import partial
from shapely.geometry import LineString, Point, Polygon, MultiPoint, MultiPolygon, MultiLineString
from operator import is_not
import pyogrio.errors
from tqdm import tqdm
from pandas import Series, concat, DataFrame
from geopandas import read_file
from numpy import array, where, argmax
import sys
import os
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ..utils import discover_docs, DataEng, ReMethods, GenRegex


class EvalChecks:

    def __init__(self):

        # Key => main label / binary raster value; value => alternative geologic names
        # Feature Type accepted values: Point, LineString, or Multiple (either LineString or Point only)

        self.lne_pnt_ontolog = {
            "Pit"                               : [{"feature_type" : "Point", "values" : ["Pit", "1_pt"]}],
            "Mine Shaft"                        : [{"feature_type" : "Point", "values" : ["Mine Shaft", "2_pt", "shaft"]}],
            "Prospect"                          : [{"feature_type" : "Point", "values" : ["3_pt", "prospect pit", "prospect"]}],
            "Mine Tunnel"                       : [{"feature_type" : "Point", "values" : ["Mine Tunnel", "4_pt", "adit"]}],
            "Quarry"                            : [{"feature_type" : "Point", "values" : ["5_pt", "1_pt", "quarry"]}],
            "Inclined Bedding"                  : [{"feature_type" : "Point", "values" : ["strike and dip", "bedding", "bedding_pt",
                                                                                          "bedding_inclined_pt", "inclined bedding"]}],
            "Overturned Bedding"                : [{"feature_type" : "Point", "values" : ["overturned", "bedding_overturned_pt"]}],
            "Horizontal Bedding"                : [{"feature_type" : "Point", "values" : ["horiz bedding", "horizontal bedding", "flat bedding"]}],
            "Vertical Bedding"                  : [{"feature_type" : "Point", "values" : ["vertical", "bedding vertical"]}],
            "Bedding and Foliation"             : [{"feature_type" : "Point", "values" : ["bedding and foliation"]}],
            "Bedding"                           : [{"feature_type" : "Point", "values" : ["bedding", "bedding w/tops", "crumpled bedding"]}],
            "Inclined Foliation"                : [{"feature_type" : "Point", "values" : ["foliation", "inclined metamorphic foliation"]}],
            "Inclined Foliation Igneous"        : [{"feature_type" : "Point", "values" : ["igneous foliation", "inclined flow banding", "inclined igneous"]}],
            "Vertical Foliation"                : [{"feature_type" : "Point", "values" : ["vert foliation", "foliation vertical"]}],
            "Foliation"                         : [{"feature_type" : "Point", "values" : ["foliation"]}],
            "Lineation"                         : [{"feature_type" : "Point", "values" : ["lineation pt", "plunge direction"]}],
            "Vertical Joint"                    : [{"feature_type" : "Point", "values" : ["joint vertical"]}],
            "Sink Hole"                         : [{"feature_type" : "Point", "values" : ["sink hole"]}],
            "Drill Hole"                        : [{"feature_type" : "Point", "values" : ["drillhole", "well pt", "Saprolite_pt", "paleomag_reversed_pt"]}],
            "Gravel Pit"                        : [{"feature_type" : "Point", "values" : ["1_pt", "Pit"]}],
            "Inclined Metamorphic"              : [{"feature_type" : "Point", "values" : ["inclined_foliation_pt"]}],
            "Inclined Foliation Metamorphic"    : [{"feature_type" : "Point", "values" : ["inclined foliation", "inclined metamorphic", "metamorphic foliation"]}],
            "Inclined Flow Banding"             : [{"feature_type" : "Point", "values" : ["inclined_foliation_igneous"]}],
            "Drill Hole Filled"                 : [{"feature_type" : "Point", "values" : ["loc_strat_section_pt", "Crystalline_rocks_pt", "well_pt",
                                                                                          "coll_breccia_pipe_pt", "sample_locality_pt", "drill hole filled"]}],
            #"GeochronPoint"                     : [{"feature_type" : "Point", "values" : ["GeochronPoint"]}],
            #"Shear Zone"                        : [{"feature_type" : "Point", "values" : ["Shear Zone"]}],
            "Fault"                        : [{"feature_type" : "LineString", "values" : ["fault_line", "normal_fault_line", "left_lat_slip_fault_line",
                                                                                          "strike_slip_fault_line", "wrench_fault_line", "hi_angle_fault_line",
                                                                                          "lo_angle_fault_line", "nrlssfault_line", "strike-slip_fault_line",
                                                                                          "generic_fault_line", "normal_rl_ss_fault_line", "wrfault_line",
                                                                                          "sinistral_strike-slip_fault_line", "reverse", "inferred", "normal fault"]}],
            "Thrust Fault"                 : [{"feature_type" : "LineString", "values" : ["Thrust fault line"]}],
            "Contour"                      : [{"feature_type" : "LineString", "values" : ["Contour_line"]}],
            #"Fold"                         : [{"feature_type" : "LineString", "values" : ["anticline", "monocline", "syncline", "antiform", "synform"]}],
            #"Vein"                         : [{"feature_type" : "LineString", "values" : ["Vein"]}],
        }

    def _get_id(self, value):
        try:
            return value.split(self.path)[-1].split("\\")[1:][0].replace("//", "").replace("\\", "").replace("/", "")
        except IndexError:
            return None

    def check_file_ext(self, path: str, file_ext: str):
        file_df = (
            discover_docs(path = path)
            .assign(file_ext=lambda d: d[['filename']].apply(lambda e: os.path.splitext(*e)[-1], axis=1),
                    has_file=lambda d: d['file_ext'].str.contains(file_ext))
        )

        return file_df

    def check_file_types(self, path: str, file_ext: str):

        self.path = path

        file_df = (
            discover_docs(path = path)
            .assign(id       = lambda d: d[['directory']].apply(lambda e: self._get_id(*e), axis=1),
                    file_ext = lambda d: d[['filename']].apply(lambda e: os.path.splitext(*e)[-1], axis=1),
                    has_file = lambda d: d['file_ext'].str.contains(file_ext))
            .query('id.notna()', engine='python')
            #.assign(id       = lambda d: d[['directory']].apply(lambda e: DataEng.last_folder_from_dir(*e), axis=1),
            #        file_ext = lambda d: d[['filename']].apply(lambda e: os.path.splitext(*e)[-1], axis=1),
            #        has_file = lambda d: d['file_ext'].str.contains(file_ext))
        )

        return file_df

    def check_missing(self, current_set: List, new_set) -> List:
        return list(set(new_set) - set(current_set))

    def unmatch(self, current_set: List, new_set):
        return list(set(current_set) - set(new_set))

    def geologic_name(self, split_tif_by, value):
        return ' '.join(value.split(split_tif_by)[1:-1])

    def feature_type(self, split_tif_by, tif_ext_file, value):
        return value.split(split_tif_by)[-1].split(tif_ext_file)[0]

    def prep_shp_data(self, path: str,  ngmdb_id: Union[str, int], split_tif_by: str = "_", shp_ext: str = ".shp", tif_ext: str = ".tif"):

        #self.split_tif_by = split_tif_by
        #self.tif_ext_file = tif_ext

        shp_data = (
            self.check_file_types(path = path, file_ext = shp_ext)
            .query('id == @ngmdb_id')
            .pipe(lambda a: concat([a[a['file_ext'] == shp_ext],
                                    a[a['file_ext'] == tif_ext].assign(geologic  = lambda b: b[['filename']].apply(lambda e: self.geologic_name(split_tif_by, *e), axis=1),
                                                                       feat_type = lambda b: b[['filename']].apply(lambda e: self.feature_type(split_tif_by, tif_ext, *e), axis=1))]))
            [['path', 'filename', 'file_ext', 'geologic', 'feat_type']]
        )

        return shp_data

    def binary_geologic_data(self, path, ngmdb_id):
        # Prepare shapefile and tif file list per ID
        shp_binary_data = self.prep_shp_data(path = path, ngmdb_id = ngmdb_id).reset_index()

        # Filter by shapefile
        shp_data        = shp_binary_data.query('file_ext == ".shp"')

        # Filter by tif file and clean out improper geologic values using regex
        binary_data = (
            shp_binary_data
            .query('file_ext == ".tif"')
            .assign(clean_geologic = lambda d: d[['geologic']].apply(lambda e: re.split('\\w{1,99}[-]\\d{1,99}\\s{1,99}\\d{1,99}\\s{1,99}', *e)[-1], axis=1))
        )

        return [shp_data, binary_data]

    def _sequence_matcher(self, str1, str2_list, threshold):
        seq_match = ReMethods().max_sequence_matcher(str1      = str1,
                                                     str2_list = str2_list)

        if seq_match[1] > threshold:
            return [seq_match[0], seq_match[1], str1]

        else:
            return None

    def _cosine_matcher(self, str1, str2_list, threshold):
        pre_cos_match = []
        for t in str2_list:
            try:
                pre_cos_match.append([str1, t, DataEng.cosine_similarity(data=[str1, t])[0][0]])
            except (TypeError, ValueError):
                pass

        if len(pre_cos_match) > 0:
            cos_df  = DataFrame(pre_cos_match, columns=['unique_geo_value', 'cand_geo_value', 'cos_value'])
            val_max = cos_df['cos_value'].max()
            if val_max > threshold:
                cos_df = cos_df.query('cos_value == @val_max')
                return [cos_df['cand_geo_value'].iloc[0], val_max, str1]

            else:
                return None

        else:
            return None

    def _key_value_matcher(self, candidate_values, matcher_type, threshold, onto_keys, onto_values = Union[None, List]) -> Union[List, None]:

        if matcher_type == "sequence-ontology_key":

            tmp_match = list(filter(partial(is_not, None),
                                    [self._sequence_matcher(str1      = k,
                                                            str2_list = candidate_values,
                                                            threshold = threshold)
                                     for k in onto_keys]))

            if len(tmp_match) > 0:
                return self._construct_df_match(tmp_match = tmp_match)

            else:
                return None

        elif matcher_type == "cosine-ontology_key":
            tmp_match = list(filter(partial(is_not, None),
                                    [self._cosine_matcher(str1      = k,
                                                          str2_list = candidate_values,
                                                          threshold = threshold)
                                    for k in onto_keys]))

            if len(tmp_match) > 0:
                return self._construct_df_match(tmp_match = tmp_match)

            else:
                return None

        elif matcher_type == "sequence-ontology_value":
            tmp_match = list(filter(partial(is_not, None),
                                    [self._sequence_matcher(str1      = v,
                                                            str2_list = candidate_values,
                                                            threshold = threshold)
                                     for k in onto_keys for v in onto_values[k]]))

            if len(tmp_match) > 0:
                return self._construct_df_match(tmp_match = tmp_match)

            else:
                return None

        elif matcher_type == "cosine-ontology_value":
            tmp_match = list(filter(partial(is_not, None),
                                    [self._cosine_matcher(str1      = v,
                                                          str2_list = candidate_values,
                                                          threshold = threshold)
                                     for k in onto_keys for v in onto_values[k]]))

            if len(tmp_match) > 0:
                return self._construct_df_match(tmp_match = tmp_match)

            else:
                return None

    def _construct_df_match(self, tmp_match: List):
        tmp_df = DataFrame(tmp_match, columns=['tmp_value', 'threshold', 'key_value'])
        id_max = tmp_df['threshold'].max()
        tmp_df = tmp_df.query('threshold == @id_max').iloc[0]

        return [tmp_df['tmp_value'], tmp_df['threshold']]

    def _check_valid_instance(self, match_list, field, match_method, geom_type):
        if isinstance(match_list, list):
            self.tmp_match.append(match_list[0])
            self.val_match.append(match_list[1])
            self.field_match.append(field)
            self.match_method.append(match_method)
            self.feature_type.append(geom_type)
            if self.accept_dict is True:
                self.dict_append.append(self.dict_val)
            else:
                self.dict_append.append(None)

            return True

        else:
            return False

    def match_geologic_value_to_field_shp(self, shp_files: Union[Series, List], unique_geologic: List, threshold: float = 0.7):
        onto_keys             = list(self.lne_pnt_ontolog.keys())
        fin_match             = []
        other_geologic_values = [] # to capture other geologic values - only when there are matches
        for shps in shp_files:
            try:
                # Read shapefile, acquire geometry field and its type
                shp           = read_file(shps)
                geom_field    = shp.geometry.name
                get_geom_type = shp[geom_field].iloc[0]
                shp           = shp.select_dtypes(exclude = ['geometry', 'int', 'int32', 'int64',
                                                             'float', 'float32', 'float64', 'datetime64'])

                shp_fields = dict(shp.dtypes)
                field_keys = list(shp_fields.keys())

                # Iterate through each unique value - i.e., from the binary raster
                for u in unique_geologic:

                    self.tmp_match, self.field_match, self.val_match, \
                    self.dict_append, self.match_method, self.feature_type, \
                    incorp_other_geo_values = [],[],[],[],[],[],[]

                    # Iterate through each field & acquire its candidate values
                    for f in field_keys:
                        try:
                            pre_cand_values  = [val.lower() for val in list(shp[f].unique())]
                            self.dict_val    = {val.lower() : val for val in list(shp[f].unique())}
                            self.accept_dict = True
                        except AttributeError:
                            pre_cand_values  = list(shp[f].unique())
                            self.accept_dict = False

                        # Lines and Points are tricky - we will need to use ontology and find best matches
                        if isinstance(get_geom_type, LineString) or isinstance(get_geom_type, MultiLineString) or \
                                isinstance(get_geom_type, Point) or isinstance(get_geom_type, MultiPoint):

                            clean_pre_cand_values = filter(lambda x: x is not None, pre_cand_values)
                            # Direct match to the ontology binary keys - if no match then it indicates that the binary rasters have not been
                            # spelled correctly or it does not exist.
                            if u[1] in onto_keys:
                                acq_ont_values = self.lne_pnt_ontolog[u[1]]
                                # If there is more than 1 feature type with associated values in the list, iterate.
                                for a in acq_ont_values:
                                    geom_type  = a['feature_type']

                                    # Retain the specific geometry type whenever a geologic value can be multiple
                                    if geom_type == "Multiple":
                                        geom_type == type(get_geom_type).__name__

                                    # Clean up using regex
                                    ont_values = [re.sub(r'\W+', ' ', a).replace("_", " ").lower() for a in a['values']]
                                    pre_values = [re.sub(r'\W+', ' ', p).replace("_", " ").lower() for p in clean_pre_cand_values]

                                    # Naive comparison approach - direct match (and preferred)
                                    naive_inter = list(set(ont_values).intersection(set(pre_values)))
                                    if len(naive_inter) > 0:
                                        pre_cand_value = [pre_cand_values[pre_values.index(n)] for n in naive_inter]
                                        for p in pre_cand_value:
                                            self._check_valid_instance(match_list   = [p, 1.0],
                                                                       field        = f,
                                                                       match_method = "naive-regex-ontology",
                                                                       geom_type    = geom_type)
                                            incorp_other_geo_values.append(DataFrame({"binary_value" : [u[1]],
                                                                                      "cand_values"  : [pre_cand_values],
                                                                                      "field"        : [f],
                                                                                      "geom_type": [geom_type],
                                                                                      "shp_file"     : [shps]}))
                                        break

                                    # Other approaches
                                    else:
                                        # Sequence
                                        seq_ont_list = [self._sequence_matcher(str1      = o,
                                                                               str2_list = p,
                                                                               threshold = threshold)
                                                        for o in ont_values for p in pre_values]

                                        seq_ont_list = list(filter(lambda x: x is not None, seq_ont_list))

                                        if len(seq_ont_list) > 0:
                                            pre_cand_value = [[pre_cand_values[pre_values.index(s[0])], s[1]] for s in seq_ont_list]

                                            [self._check_valid_instance(match_list   = p,
                                                                        field        = f,
                                                                        match_method = 'sequence-regex-ontology',
                                                                        geom_type    = geom_type)
                                             for p in pre_cand_value]

                                            incorp_other_geo_values.append(DataFrame({"binary_value" : [u[1]],
                                                                                      "cand_values"  : [pre_cand_values],
                                                                                      "field"        : [f],
                                                                                      "shp_file"     : [shps]}))

                                            break

                                        # Computationally slow due to NLP method in use for cosine similarity - last resort
                                        #else:
                                            # Cosine
                                        #    cos_ont_list = [self._cosine_matcher(str1      = o,
                                        #                                         str2_list = p,
                                        #                                         threshold = threshold)
                                        #                    for o in ont_values for p in pre_values]

                                        #    cos_ont_list = list(filter(lambda x: x is not None, cos_ont_list))

                                        #    if len(cos_ont_list) > 0:
                                        #        pre_cand_value = [[pre_cand_values[pre_values.index(p[0])], p[1]] for p in cos_ont_list]
                                        #        [self._check_valid_instance(match_list   = p,
                                        #                                    field        = f,
                                        #                                    match_method = 'cosine-regex-ontology',
                                        #                                    geom_type    = geom_type)
                                        #         for p in pre_cand_value]

                                        #        incorp_other_geo_values.append(DataFrame({"binary_value" : [u[1]],
                                        #                                                  "cand_values"      : [pre_cand_values],
                                        #                                                  "field"            : [f],
                                        #                                                  "shp_file"         : [shps]}))

                                        #        break

                        # Matching methods for polygons - names don't change as these are "geologic" values.
                        elif isinstance(get_geom_type, Polygon) or isinstance(get_geom_type, MultiPolygon):
                            # Expectation is the current binary value is not to be matching the ontology keys
                            if u[1] not in onto_keys:
                                try:
                                    seq_match = self._sequence_matcher(str1      = u[0],
                                                                       str2_list = pre_cand_values,
                                                                       threshold = threshold)

                                    seq_valid = self._check_valid_instance(match_list   = seq_match,
                                                                           field        = f,
                                                                           match_method = "sequence",
                                                                           geom_type    = "Polygon")

                                    if seq_valid:
                                        incorp_other_geo_values.append(DataFrame({"binary_value" : [u[1]],
                                                                                  "cand_values"  : [pre_cand_values],
                                                                                  "field"        : [f],
                                                                                  "geom_type"    : ["Polygon"],
                                                                                  "shp_file"     : [shps]}))
                                        break

                                    else:
                                        regex_pre_cand_values = [re.sub(r'\W+', '', p).replace("_", " ") for p in pre_cand_values]
                                        geo_value             = re.sub(r'\W+', '', u[0]).replace("_", " ")
                                        reg_seq_match         = self._sequence_matcher(str1      = geo_value,
                                                                                       str2_list = regex_pre_cand_values,
                                                                                       threshold = threshold)

                                        if reg_seq_match:
                                            pre_cand_value  = pre_cand_values[regex_pre_cand_values.index(reg_seq_match[0])]
                                            reg_seq_match   = [pre_cand_value, reg_seq_match[1]]

                                        reg_seq_valid         = self._check_valid_instance(match_list   = reg_seq_match,
                                                                                           field        = f,
                                                                                           match_method = 'sequence-regex',
                                                                                           geom_type    = "Polygon")

                                        if reg_seq_valid:
                                            incorp_other_geo_values.append(DataFrame({"binary_value" : [u[1]],
                                                                                      "cand_values"  : [pre_cand_values],
                                                                                      "field"        : [f],
                                                                                      "geom_type": ["Polygon"],
                                                                                      "shp_file"     : [shps]}))
                                            break

                                        # Computationally slow method due to NLP process - last resort
                                        #else:
                                        #    cos_match = self._cosine_matcher(str1      = u[0],
                                        #                                     str2_list = pre_cand_values,
                                        #                                     threshold = threshold)

                                        #    cos_valid = self._check_valid_instance(match_list   = cos_match,
                                        #                                           field        = f,
                                        #                                           match_method = "cosine",
                                        #                                           geom_type    = "Polygon")

                                        #    if cos_valid:
                                        #        incorp_other_geo_values.append(DataFrame({"binary_value" : [u[1]],
                                        #                                                  "cand_values"  : [pre_cand_values],
                                        #                                                  "field"        : [f],
                                        #                                                  "geom_type"    : ["Polygon"],
                                        #                                                  "shp_file"     : [shps]}))
                                        #        break

                                except TypeError:
                                    pass

                    # If there are 1 or more matches append results by keeping the one with absolute highest match
                    if len(self.val_match) > 0:
                        max_idx      = self.val_match.index(max(self.val_match))
                        get_val      = self.tmp_match[max_idx]
                        field_val    = self.field_match[max_idx]
                        match_method = self.match_method[max_idx]
                        geom_type    = self.feature_type[max_idx]
                        filt_dict    = list(filter(None, self.dict_append))
                        if len(filt_dict) > 0:
                            get_dict = self.dict_append[max_idx][get_val]
                            fin_match.append([u[1], field_val, get_dict, self.val_match[max_idx], shps,
                                              match_method, geom_type, "no_corrupt"])
                            other_geologic_values.append(concat(incorp_other_geo_values))
                        else:
                            fin_match.append([u[1], field_val, get_val, self.val_match[max_idx], shps,
                                              match_method, geom_type, "no_corrupt"])
                            other_geologic_values.append(concat(incorp_other_geo_values))

            except pyogrio.errors.DataSourceError:
                print(f"Failed to open {shps} due to data corruption / insufficient extension requirements.")
                fin_match.append([None, None, None, None, shps, None, None, "corrupted"])

        return [fin_match, other_geologic_values]

    def regex_update_match(self, path: str, id_list, file_ext):

        file_df = (
            discover_docs(path = path)
            [['directory']]
            .drop_duplicates(['directory'], keep='first')
        )

        list_files = list(file_df['directory'].unique())
        fin_match  = []
        for g in tqdm(id_list):
            tmp_re    = GenRegex(string = g).regex_str
            tmp_match, file_match = [],[]
            for l in list_files:
                re_match = re.search(tmp_re, string = l)
                if re_match is not None:
                    start = re_match.regs[0][0]
                    end   = re_match.regs[0][1]
                    match = l[start:end]
                    tmp_match.append(match)
                    file_match.append(l)

            if len(tmp_match) > 0:
                clean_match = array([ReMethods().max_sequence_matcher(str1 = g, str2_list = t)[1]
                                     for t in tmp_match])
                get_max     = argmax(clean_match)

                if clean_match[get_max] == 1:
                    clean_match = array(clean_match)
                    tmp_match   = array(tmp_match)
                    file_match  = array(file_match)
                    mult_vals   = where(clean_match == 1.0)

                    for m in mult_vals[0]:
                        fin_match.append([g, list(tmp_match)[m], list(file_match)[m]])

                else:
                    fin_match.append([g, None, None])

        if len(fin_match) > 0:

            match_df = (
                DataFrame(fin_match, columns=['id', 'match_value', 'directory'])
                .merge(discover_docs(path = path), on=['directory'])
                .assign(file_ext = lambda d: d[['filename']].apply(lambda e: os.path.splitext(*e)[-1], axis=1),
                        has_file = lambda d: d['file_ext'].str.contains(file_ext))
            )

            return match_df

        else:
            return None
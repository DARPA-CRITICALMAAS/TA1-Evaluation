"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer.

Date Created: Sept. 2024

Date Modified: Oct. 2024

About: Evaluating Georeferencing from the performer's narrow AI models.
       Function for parallel processing that parses out the GCPs to its corresponding system and system version, parses
       and creates GroundControlPoints (GCPs), uses the GCPs to create an AffineTransformer, and if all of these
       requirements have been met then to perform RMSE via Geodesic distance in km.

Expected Output(s): DataFrame that can be exported preferably as a feather or parquet file.
    - Schema:
        - "cog_id"              => COG ID
        - "annotated_system"    => Name of the annotated system
        - "annotated_version"   => Annotated system version
        - "performer"           => Name of the performer system
        - "performer_version"   => Performer system version
        - "log"                 => Log of what happened during eval.
        - "rmse"                => RMSE score.
        - "geodesic_pnts"       => Geodesic distance per point - list form.
        - "annotated_pnts"      => Annotated points (i.e., randomly selected pixel space to coordinates converted from
                                    its AffineTransformer) - list form.
        - "performer_pnts"      => Performer points (i.e., randomly selected pixel space to coordinates converted from
                                    its AffineTransformer) - list form.
        - "sample_pnts"         => Number of randomly selected pixels to run for RMSE.

Pre-requisites:
    - Must run PreEvalGeoref class to acquire the DataFrame required for this workflow.
    - Token to access the CDR.
    - Basic understanding regards to the formatting of the GCPs in the CDR. Try out a COG ID to view the format at
      https://api.cdr.land, specifically the /v1/maps/cog/gcps/{cog_id} rest endpoint.
    - Communicate with the performers and annotator of what is their system and system version you should be extracting.
"""
# Data Engineering packages
from pandas import DataFrame, read_csv, concat
from geopy.distance import geodesic as GD
import numpy as np
import re

# Miscellaneous tools
from typing import List, Union
from functools import partial
from multiprocessing import Manager, cpu_count

# OS and Sys packages to import custom packages outside its relative directory.
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Custom packages to import
from ..utils import discover_docs, DataEng, SpatialOps, ParallelPool
from ..cdr import GCPS, CDR2Data, GeorefConvert


class GeorefEval:

    def __init__(self, cdr_token: str, georef_dataset: Union[str, DataFrame], output_dir: str, cdr_systems: dict):

        """
        :param cdr_token: str -> Token to have authorization access to the CDR.
        :param georef_dataset: DataFrame or csv file (as a str) that contains id, cog_id, tif_file, pixel space information, random
                               pixels for evaluation, and TIF file information derived from the georeferencing ground truth set.
        :param output_dir: str -> Main output directory to store the GCP results from the performers.
        :param cdr_systems: dict -> Used to help parse out the GCP data pulled from the CDR. The values of the annotated
                                    key must be a dictionary with "system" and "system_version" as the nested keys.
                                    The values of the performers key must be a list representing each performer with a
                                    nested dictionary. "system" and "system_version" as the nested keys per list item.

                                    Please see below an example format:

                                {
                                    "annotated"  : {"system" : "ngmdb", "system_version" : "2.0"},
                                    "performers" : [
                                                    {"system" : "uncharted-georeference", "system_version" : "0.0.5"},
                                                    {"system" : "umn-usc-inferlink", "system_version" : "0.0.7"}
                                    ]
                                }
        """

        self.cdr_systems = cdr_systems
        self.performers  = self.cdr_systems['performers'] # Retrieve performer related information
        self.annotated   = self.cdr_systems['annotated'] # Retrieve annotated related information

        # If the georeference data is a string (aka csv file), then read it first
        if isinstance(georef_dataset, str):
            georef_dataset = read_csv(georef_dataset)

        self.georef_dataset = georef_dataset
        self.cog_ids        = self.georef_dataset['cog_id'].unique() # Acquire list of COG IDs

        # Parallel Thread - Download results from the CDR using the /v1/maps/cog/gcps/{cog_id} rest endpoint.
        config = {
            "token"      : cdr_token,
            "cog_ids"    : self.cog_ids,
            "output_dir" : output_dir
        }

        gcp_download_run     = GCPS(config = config)
        self.failed_download = gcp_download_run.failed_cogs

        # Identify the GCPs files that have been downloaded and match its naming convention to the COG ID for identity
        # and eval. purposes. The naming convention should have part of its directory named after the COG ID during GCPS.
        self.gcp_files = (
            discover_docs(path = output_dir)
            .assign(is_file = lambda d: d['filename'].str.contains("_gcps.json"))
            .query('is_file == True')
            [['directory', 'path', 'filename']]
            .pipe(lambda a: a.merge(self._match_gcp_file_to_cog(df = a), on=['directory']))
        )

        # Prepare parallel processing of the Georeferencing evaluation.
        gcp_cogs     = concat([DataFrame([[c, self.gcp_files.query("cog_id == @c"), georef_dataset.query("cog_id == @c")]],
                                  columns=['cog_id', 'gcp_file_dir', 'georef_dataset'])
                        for c in self.cog_ids])
        split_data   = np.array_split(gcp_cogs, cpu_count())
        L1           = Manager().list()
        partial_func = partial(self._main, L1=L1)
        ParallelPool(start_method="spawn", partial_func=partial_func, main_list=split_data)
        self.concat_df = concat(L1) # RMSE output as the final - to view and aggregate

    def _match_gcp_file_to_cog(self, df) -> DataFrame:
        """
        Match the GCP result file to the COG ID based on its naming convention.

        :return: DataFrame of the COG ID and its directory.
        """
        cog_match = []
        for c in self.cog_ids:
            for p in df['directory']:
                if re.search(pattern = c, string = p):
                    cog_match.append([c, p])
                    break

        return DataFrame(cog_match, columns=["cog_id", "directory"])

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

    def _gcps_df(self, parse_systems: DataFrame, system_data) -> List:
        """
        Filter the parsed GCP DataFrame based on its system data (i.e., system --> e.g., @annotated or @performer value)
        and assign GCPS by parsing it out and creating an actual GroundControlPoint via rasterio.

        :param parse_systems: Parsed GCP DataFrame.
        :param system_data: System data - performer or annotated.

        :return: List --> Filtered parsed GCP system with
        """

        # Filter the parsed GCP data based on its system data & assign GCPs
        gcps_df = (
            parse_systems
            .query('system == @system_data')
            .assign(gcps=lambda a: a[['data']].apply(lambda e: self._parse_gcps(*e), axis=1))
        )

        # If the subset DataFrame exists - then perform AffineTransformer
        if len(gcps_df) > 0:
            # AffineTransformer from a list of GCPs.
            try:
                transformer = SpatialOps().affine_transformer_from_gcps(gcps_list = gcps_df['gcps'].tolist())
            except TypeError:
                transformer = False # Set to False indicating a failed transformer
        else:
            transformer = None # Set to None as there is no data from the subset to construct GCPs.

        return [gcps_df, transformer]

    def _cdr_match_system(self, dict_value: dict, performer: dict) -> List:
        """
        Part of the _main function - parse through the GCP dictionary and tie the extraction to its matched system and
        its system version.

        :param dict_value: Dictionary of the GCP.
        :param performer: Dictionary of the performer information.

        :return: List containing the system, system version, and GCP dictionary value - if applicable.
        """
        sys_performer   = performer['system']
        sys_v_performer = performer['system_version']
        sys_annotated   = self.annotated['system']
        sys_v_annotated = self.annotated['system_version']

        # If the dictionary GCP matches one of the performers and its system version
        if (dict_value['system'] == sys_performer) and (dict_value['system_version'] == sys_v_performer):
            return [sys_performer, sys_v_performer, dict_value]

        # If the dictionary GCP matches to the annotated system and its system version
        elif (dict_value['system'] == sys_annotated) and (dict_value['system_version'] == sys_v_annotated):
            return [sys_annotated, sys_v_annotated, dict_value]

        # Otherwise no match or system and system version mismatch.
        else:
            return [None, None, None]

    def _rmse(self, annotated_transformer, result_transformer, cog_data) -> List:
        """
        Perform RMSE via Geodesic distance in km.

        :param annotated_transformer: Annotated AffineTransformer.
        :param result_transformer: Result AffineTransformer.
        :param cog_data: From the georeferenced dataset, contains the COG ID information including randomly selected pixel spaces.

        :return: List containing the RMSE and the geodesic distance per random selected point (for granularity).
        """


        pnts = [[annotated_transformer.xy(rows=h, cols=w),
                 result_transformer.xy(rows=h, cols=w)]
                for h, w in zip(cog_data['random_pix_height'], cog_data['random_pix_width'])]

        # To rearrange to latitude and longitude as opposed to longitude and latitude
        pnts           = [[(p[0][1], p[0][0]), (p[1][1], p[1][0])] for p in pnts]

        geodesic_info  = [[GD(list(p[0]), list(p[1])).km, p[0], p[1]] for p in pnts]
        geodesic_pnts  = [g[0] for g in geodesic_info]
        annotated_pnts = [g[1] for g in geodesic_info]
        performer_pnts = [g[2] for g in geodesic_info]
        rmse           = np.sqrt(np.square(geodesic_pnts).mean())

        return [rmse, geodesic_pnts, annotated_pnts, performer_pnts]
        #return pnts

    def _main(self, cog_data: DataFrame, L1):
        """
        Function for parallel processing that parses out the GCPs to its corresponding system and system version,
        parses and creates GroundControlPoints (GCPs), uses the GCPs to create an AffineTransformer, and if all of these
        requirements have been met then to perform RMSE via Geodesic distance in km.

        :param cog_data: A nested list that has been split into equal number to devote per CPU. Each nested list items
                         contains - COG ID, its corresponding GCP file directory information (to read), and
        :param L1: List manager to append results during parallel processing.
        """

        # Acquire unique performers & the annotated system for parsing and filtering process.
        unique_performers  = [[p['system'], p['system_version']] for p in self.performers]
        performer_systems  = [u[0] for u in unique_performers]
        performer_versions = [u[1] for u in unique_performers]
        #annotated_system   = self.annotated['system']
        #annotated_version  = self.annotated['system_version']

        tmp_data = []
        for cog_info in range(len(cog_data)):
            tmp_row = cog_data.iloc[cog_info]
            cog_id  = tmp_row['cog_id']
            cog_gcp = tmp_row['gcp_file_dir']
            cog_df  = tmp_row['georef_dataset']

            cog_result = cog_gcp['path'].iloc[0] # the filename of the GCP result.

            for a in self.annotated:
                parse_systems = CDR2Data(performers = self.performers,
                                         annotated  = a).gcp_parse_cdr_data(cdr_file = cog_result)

                annotated_system  = a['system']
                annotated_version = a['system_version']

                if len(parse_systems) > 0:
                    annotated_tmp = parse_systems.query('system == @annotated_system and system_version == @annotated_version')
                    if len(annotated_tmp) > 0:
                        break


            # Check to make sure there is data from the CDR to continue metric evaluation.
            if len(parse_systems) > 0:

                # Group systems (including performer) check - to make sure there is annotated and performer data to evaluate.
                grp_chck       = parse_systems.groupby(['system'], as_index=False).count()
                performer_chck = grp_chck.query('system in @performer_systems')
                annotated_chck = grp_chck.query('system == @annotated_system')

                # What we want to have - performer result info and annotated info at the same time.
                if len(performer_chck) > 0 and len(annotated_chck) > 0:

                    # Filter GCP information that is from the annotated system
                    #annotated_data        = self._gcps_df(parse_systems = parse_systems,
                    #                                      system_data   = annotated_system)
                    annotated_data        = GeorefConvert().from_gcp_data(parse_systems = parse_systems,
                                                                          system_data   = annotated_system)
                    annotated_df          = annotated_data[0]
                    annotated_transformer = annotated_data[1]

                    # If annotated parsed system exists and the transformer is not None.
                    if len(annotated_df) > 0 and annotated_transformer is not None:

                        # If the transformer is valid.
                        if annotated_transformer is not False:

                            # Go through each unique performer to filter
                            for u in unique_performers:

                                # Filter GCP information
                                performer_data     = self._gcps_df(parse_systems = parse_systems,
                                                                   system_data   = u[0])
                                performer_df       = performer_data[0]
                                result_transformer = performer_data[1]

                                # If performer parsed system exists and the transformer is not None.
                                if len(performer_df) > 0 and result_transformer is not None:

                                    # If the transformer is valid then perform RMSE
                                    if result_transformer is not False:

                                        geodesic_rmse  = self._rmse(annotated_transformer = annotated_data[1],
                                                                    result_transformer    = result_transformer,
                                                                    cog_data              = cog_df)

                                        rmse           = geodesic_rmse[0]
                                        geodesic_pnts  = geodesic_rmse[1]
                                        annotated_pnts = geodesic_rmse[2]
                                        performer_pnts = geodesic_rmse[3]

                                        #tmp_data.append([cog_id, annotated_system, annotated_version, u[0], u[1],
                                        #                 "success", rmse])
                                        tmp_data.append([cog_id, annotated_system, annotated_version, u[0], u[1],
                                                         "success", rmse, geodesic_pnts, annotated_pnts,
                                                         performer_pnts, len(geodesic_pnts)])

                                    # Bad transformer from the GCP format output by the performer.
                                    else:
                                        tmp_data.append([cog_id, annotated_system, annotated_version, u[0], u[1],
                                                         "performer-Bad Transformer / Incompatible GCP format",
                                                         None, None, None, None, None])

                                # Performer missing data
                                else:
                                    tmp_data.append([cog_id, annotated_system, annotated_version, u[0], u[1],
                                                     "performer-missing", None, None, None, None, None])

                        # Bad transformer from the GCP format output by the annotator.
                        else:
                            tmp_data.append([cog_id, annotated_system, annotated_version, None, None,
                                             "annotated-Bad Transformer / Incompatible GCP format", None, None, None,
                                             None, None])

                    # Missing annotated GCP data.
                    else:
                        tmp_data.append([cog_id, annotated_system, annotated_version, None, None,
                                         "annotated-missing-GCP-parsing", None, None, None, None, None])

                # Indicates that there is only annotated data, but no performer result - either CDR has not been updated or tool fail.
                elif len(annotated_chck) > 0 and len(performer_chck) == 0:
                    tmp_data.append([cog_id, annotated_system, annotated_version, None, None, "performer-missing", None,
                                     None, None, None, None])

                # Indicates that the tools have ran and the CDR has updated, but there is no annotated data to cross-check.
                elif len(performer_chck) > 0 and len(annotated_chck) == 0:
                    tmp_data.append([cog_id, None, None, None, None, "annotated-missing", None,
                                     None, None, None, None])

            # Dead-end - there is no data to parse out
            else:
                tmp_data.append([cog_id, None, None, None, None, "annotated-performer-missing",
                                 None, None, None, None, None])

        L1.append(DataFrame(tmp_data, columns=["cog_id", "annotated_system", "annotated_version", "performer",
                                               "performer_version", "log", "rmse", "geodesic_pnts",
                                               "annotated_pnts", "performer_pnts", "sample_pnts"]))
        #L1.append(DataFrame(tmp_data, columns=['cog_id', 'annotated_systems', 'annotated_version', 'performer',
        #                                       'performer_version', 'log', 'pre_rmse']))

"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer.

Date Created: October 2024

Date Modified: November 2024

About: Multithreading of pulling feature extraction results from the CDR based on a list of COG IDs. Current feature
       extraction REST endpoints are "polygon", "line", and "point".

Pre-requisites:
    - CDR API Token
    - List of COG IDs in csv file or as a list.
    - Up-to-date CDR - main host url and REST Endpoints.
    - Results after running TA1 (i.e., Deep Learning models ) tools to inference on geologic maps.

Warnings: Must not have security restrictions (e.g., ZScaler) or blacklist that causes 403 (forbidden) errors.

Outputs:
    - Successful download would be in the following format:
        - for polygons: "{output_dir}/{cog_id}/inferenced/polygon/{system}__{sys_v}__{cog_id}__polygon.json"
        - for lines: "{output_dir}/{cog_id}/inferenced/line/{system}__{sys_v}__{cog_id}__line.json"
        - for points: "{output_dir}/{cog_id}/inferenced/point/{system}__{sys_v}__{cog_id}__point.json"

    - Unsuccessful download(s) can be accessed either:
        - Return DataFrame (2nd item) OR
        - {output_dir}/{extraction_type}_failed_feature.csv" --> extraction_type would be "polygon", "line", or "point".
"""

# Data Engineering related packages
from json import loads
from pandas import DataFrame, concat
from .gen_cdr import Generic

# Miscellaneous packages
from tqdm import tqdm
from functools import partial
from multiprocessing import Manager
import httpx
from typing import Union, List

# Packages used to import custom-made packages outside of relative path
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Importing custom-made packages
from ..utils import DataEng, ParallelThread


class FeatExtractCDR:

    def __init__(self, config: Union[str, dict], cdr_systems: dict, fast_api_url = "https://api.cdr.land/v1"):
        """
        :param config: Dictionary or JSON config file that requires the following parameters:
            - token         => Str          => CDR token for access.
            - cog_ids       => List or Str  => CSV file containing COG IDs or a list of COG IDs.
            - cog_id_field  => Str          => Name of COG ID Field - only applies if you've listed cog_ids as a csv file.
            - output_dir    => Str          => Output directory to store the JSON files.
        :param cdr_systems: Dictionary of the performer information.

            cdr_systems = {
                "performers" : [{"system" : "uiuc-icy-resin", "system_version" : "0.4.6"}]
            }

        :param fast_api_url: Str => Fast API (i.e., host) URL to the CDR - default provided. Change the main host URL if needed.
        """

        self.fast_api_url = fast_api_url
        self.client       = httpx.Client(timeout=None)
        self.cdr_systems  = cdr_systems

        kwargs          = DataEng.read_kwargs(config = config)
        token           = kwargs['token']
        cog_ids         = kwargs['cog_ids'] # Must be either a list or a string pointing to a csv file.
        cog_id_field    = kwargs.get('cog_id_field', None) # If cog_ids points to csv file, then state
        self.output_dir = kwargs['output_dir']

        # Checks if the COG IDs are list of CSV.
        self.cog_ids = Generic.acquire_cog_ids_from_csv(cog_ids      = cog_ids,
                                                        cog_id_field = cog_id_field)

        # Construct authorization headers.
        self.headers = {"accept"        : "application/json",
                        "Authorization" : f"Bearer {token}"}

    def polygon_results(self, georeference_data: bool = True, legend_data: bool = True) -> List:
        """
        Download polygon feature extraction results from the CDR.
        :return: List of COG IDs with COG-CDR URLs and list of cog ids that failed to download from the CDR.
        """
        cog_urls    = self._build_urls(extraction_type="polygon")
        cog_info    = self._parallel(cog_urls=cog_urls,
                                     extraction_type="polygon",
                                     georeference_data=georeference_data,
                                     legend_data=legend_data)
        return [cog_urls, cog_info]

    def line_results(self, georeference_data: bool = True, legend_data: bool = True) -> List:
        """
        Download line feature extraction results from the CDR.
        :return: List of COG IDs with COG-CDR URLs and list of cog ids that failed to download from the CDR.
        """
        cog_urls = self._build_urls(extraction_type="line")
        cog_info = self._parallel(cog_urls=cog_urls,
                                  extraction_type="line",
                                  georeference_data=georeference_data,
                                  legend_data=legend_data)
        return [cog_urls, cog_info]

    def point_results(self, georeference_data: bool = True, legend_data: bool = True) -> List:
        """
        Download point feature extraction results from the CDR.
        :return: List of COG IDs with COG-CDR URLs and list of cog ids that failed to download from the CDR.
        """
        cog_urls    = self._build_urls(extraction_type="point")
        cog_info = self._parallel(cog_urls=cog_urls,
                                  extraction_type="point",
                                  georeference_data=georeference_data,
                                  legend_data=legend_data)
        return [cog_urls, cog_info]

    # Change here if the endpoint has changed
    def _build_urls(self, extraction_type: str = Union["polygon", "line", "point"]) -> List:
        """
        Build COG ID urls to extract specific feature extracted items.

        :param extraction_type: Accepted string values to identify which endpoint to pull data from the CDR.
        :return: Nested list of COG IDs and its respective URL.
        """
        if extraction_type == "polygon":
            cog_urls = [[cog, f"{self.fast_api_url}/features/{cog}/polygon_extractions"] # Change here if needed
                        for cog in tqdm(self.cog_ids)]

        elif extraction_type == "point":
            cog_urls = [[cog, f"{self.fast_api_url}/features/{cog}/point_extractions"] # Change here if needed
                        for cog in tqdm(self.cog_ids)]

        elif extraction_type == "line":
            cog_urls = [[cog, f"{self.fast_api_url}/features/{cog}/line_extractions"] # Change here if needed
                        for cog in tqdm(self.cog_ids)]

        return cog_urls

    def _main_pull(self, cog_url, L1, feature_type: str = Union["polygon", "line", "point"], georeference_data: bool = True,
                   legend_data: bool = True):
        """
        Main function to pull feature extraction results from the CDR and save as a JSON file.

        :param cog_url: COG URL including COG ID.
        :param L1: List Manager to append any failed COG IDs during the pull request process.
        :param feature_type: Feature Type - accepted string values to know what is being extracted.
        """

        georeference_data = str(georeference_data).lower()
        legend_data       = str(legend_data).lower()

        for c in self.cdr_systems['performers']:
            system      = c['system']
            sys_v       = c['system_version']

            append_data = []
            page        = 0
            success     = "success"

            while True:
                build_url = f"{cog_url[1]}?system_version={system}__{sys_v}&georeference_data={georeference_data}&legend_data={legend_data}&page={page}&size=1000"
                response  = self.client.get(build_url, headers = self.headers)

                if response.status_code == 200:
                    data = loads(response.content.decode('utf-8'))
                    if len(data) > 0:
                        append_data.append(data)
                        page += 1
                        success = success
                    else:
                        success = "success-done"
                        break
                else:
                    success = "failed"
                    break

            if len(append_data) > 0:
                output_loc = f"{self.output_dir}/{cog_url[0]}/inferenced/{feature_type}"
                output_file = f"{output_loc}/{system}__{sys_v}__{cog_url[0]}__{feature_type}.json"
                DataEng.checkdir(dir_name=output_loc)
                DataEng.write_json(output_file=output_file, data=append_data)
                L1.append(DataFrame([[cog_url[0], output_file, response.status_code,
                                      success, feature_type, system, sys_v]], columns=['cog_id', 'output_file',
                                                                                       'response_code', 'message',
                                                                                       'feature_type', 'system',
                                                                                       'sys_version']))

            # Otherwise append COG ID information that has failed
            else:
                L1.append(DataFrame([[cog_url[0], None, response.status_code,
                                      success, feature_type, system, sys_v]], columns=['cog_id', 'output_file',
                                                                                       'response_code', 'message',
                                                                                       'feature_type', 'system',
                                                                                       'sys_version']))

    def _parallel(self, cog_urls: List, extraction_type: str = Union["polygon", "line", "point"],
                  georeference_data: bool = True, legend_data: bool = True):
        """
        Parallel (multithreading) function that downloads in bulk feature extraction data from the CDR.
        :param cog_urls: Nest list of COG IDs with their respective CDR urls.
        :param extraction_type: Accepted string values to identify which endpoint to pull data from the CDR.
        :return:
        """
        L1           = Manager().list()
        partial_func = partial(self._main_pull, L1=L1, feature_type=extraction_type,
                               georeference_data=georeference_data, legend_data=legend_data)
        ParallelThread(start_method="spawn", partial_func = partial_func, main_list = cog_urls)
        concat_L1 = concat(L1)
        return concat_L1

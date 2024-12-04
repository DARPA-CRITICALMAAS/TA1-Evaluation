"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer

Date Created: August 2024

Date Modified: August 2024

About: Multithreading of pulling annotated legend items from a list of COG IDs.

Pre-requisites:
    - CDR API Token

Outputs:
    -

Warnings: Must not have security restrictions (e.g., ZScaler) that causes 403 (forbidden) errors.
"""

# Data Engineering related packages
from .gen_cdr import Generic
from json import loads
from pandas import concat, DataFrame

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

# Custom-made packages
from ..utils import DataEng, ParallelThread


class LegendItemsCDR:

    def __init__(self, config: Union[str, dict], fast_api_url="https://api.cdr.land/v1"):
        """
        :param config: Dictionary or JSON config file. The following required keyword arguments are:
            - token         => Str         => CDR API Token.
            - cog_ids       => List or Str => List of COG IDs or string pointing to a CSV file containing COG IDs.
            - cog_id_field  => Str         => Name of COG ID Field - only applies if you've listed cog_ids as a csv file.
            - output_dir    => Str         => Output path directory without trailing slash ("/").
            - validated     => Str         => Boolean string accepting either: "True" or "False".

        :param fast_api_url: Fast API URL for extraction - default value provided.
        """

        self.fast_api_url = fast_api_url
        self.client       = httpx.Client(timeout = None)

        kwargs         = DataEng.read_kwargs(config=config)
        token          = kwargs['token']
        cog_ids        = kwargs['cog_ids'] # Must be either a list or a string pointing to a csv file.
        cog_id_field   = kwargs.get('cog_id_field', None) # If cog_ids points to csv file, then state which field it is.
        output_dir     = kwargs['output_dir']
        self.validated = kwargs['validated']

        # Checks if the COG IDs are list or CSV.
        self.cog_ids = Generic.acquire_cog_ids_from_csv(cog_ids      = cog_ids,
                                                        cog_id_field = cog_id_field)

        # Construct authorization headers.
        self.headers = {"accept"        : "application/json",
                        "Authorization" : f"Bearer {token}"}

        # Build COG ID URLs.
        self.cog_urls = self._build_urls()

        # Multithreading to download
        L1           = Manager().list()
        partial_func = partial(self._main_pull, output_dir = output_dir, L1 = L1)
        ParallelThread(start_method='spawn', partial_func = partial_func, main_list = self.cog_urls)
        self.concat_L1 = concat(L1)

    def _build_urls(self) -> List:
        """
        Build COG ID urls to extract annotated legend items.

        :return: Return nested list of COG IDs and its respective URL.
        """
        cog_urls = [[cog, f"{self.fast_api_url}/features/{cog}/legend_items?validated={self.validated}"]
                    for cog in tqdm(self.cog_ids)]
        return cog_urls

    def _main_pull(self, cog_url, output_dir, L1):
        """
        Main function to pull annotated legend items from the CDR and save as a JSON file.

        :param cog_url: COG URL including COG ID.
        :param output_dir: Output directory.
        :param L1: List Manager to append any failed COG IDs during the pull request process.
        """

        response = self.client.get(cog_url[1], headers = self.headers)

        if response.status_code == 200:
            data        = loads(response.content.decode('utf-8'))
            if len(data) > 0:
                output_loc  = f"{output_dir}/{cog_url[0]}/annotated/legend"
                DataEng.checkdir(dir_name = output_loc)
                output_file = f"{output_loc}/{cog_url[0]}_legend.json"
                DataEng.write_json(output_file = output_file, data = data)
                L1.append(DataFrame([[cog_url[0], output_file, "success-done", response.status_code]], columns=['cog_id', 'output_file', 'message', 'response_code']))

            else:
                L1.append(DataFrame([[cog_url[0], None, "empty", response.status_code]], columns=['cog_id', 'output_file', 'message', 'response_code']))

        else:
            L1.append(DataFrame([[cog_url[0], None, "failed", response.status_code]], columns=['cog_id', 'output_file', 'message', 'response_code']))

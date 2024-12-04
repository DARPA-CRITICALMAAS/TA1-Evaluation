"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer

Date Created: August 2024

Date Modified: August 2024

About: Multithreading of pulling annotated GCP items from a list of COG IDs. This would be primarily used as a
       pre-requisite for georeferencing evaluation.

Pre-requisites:
    - CDR API Token
    - List of COG IDs in csv file or as a list.

Warnings: Must not have security restrictions (e.g., ZScaler) that causes 403 (forbidden) errors.

Outputs:
    - Successful download would be in the following format: "{output_dir}/{cog_id}/annotated/gcps/{cog_id}_gcps.json"
    - Unsuccessful download(s) can be accessed either:
        - "failed_cogs" variable OR
        - "{output_dir}/failed_gcps_pull.txt"
"""

# Data Engineering related packages
from json import loads
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


class GCPS:

    def __init__(self, config: Union[str, dict], fast_api_url="https://api.cdr.land/v1"):
        """
        :param config: Dictionary or JSON config file that requires the following parameters:
            - token        => Str         => CDR token for access.
            - cog_ids      => List or Str => CSV file containing COG IDs or a list of COG IDs.
            - cog_id_field => Str         => Name of COG ID Field - only applies if you've listed cog_ids as a csv file.
            - output_dir   => Str         => Output directory to store the TIF files being pulled.

        :param fast_api_url: Str => Fast API (i.e., host) URL to the CDR - default provided. Change the main host URL if needed.
        """
        self.fast_api_url = fast_api_url
        self.client = httpx.Client(timeout=None)

        kwargs       = DataEng.read_kwargs(config=config)
        token        = kwargs['token']
        cog_ids      = kwargs['cog_ids']  # Must be either a list or a string pointing to a csv file.
        cog_id_field = kwargs.get('cog_id_field', None)  # If cog_ids points to csv file, then state which field it is.
        output_dir   = kwargs['output_dir']

        # Checks if the COG IDs are list or CSV.
        self.cog_ids = Generic.acquire_cog_ids_from_csv(cog_ids      = cog_ids,
                                                        cog_id_field = cog_id_field)

        # Construct authorization headers.
        self.headers = {"accept": "application/json",
                        "Authorization": f"Bearer {token}"}

        # Build COG ID URLs.
        self.cog_urls = self._build_urls()

        # Multithreading to download
        L1 = Manager().list()
        partial_func = partial(self._main_pull, output_dir=output_dir, L1=L1)
        ParallelThread(start_method='spawn', partial_func=partial_func, main_list=self.cog_urls)
        DataEng.generic_output_text(output=f"{output_dir}/failed_gcps_pull.txt", info=str(L1))

        self.failed_cogs = L1

    def _build_urls(self) -> List:
        """
        Build COG ID urls to extract annotated GCP items.

        :return: Return nested list of COG IDs and its respective URL.
        """
        cog_urls = [[cog, f"{self.fast_api_url}/maps/cog/gcps/{cog}"] #This may require edit if the REST URL has changed.
                    for cog in tqdm(self.cog_ids)]
        return cog_urls

    def _main_pull(self, cog_url, output_dir, L1):
        """
        Main function to pull annotated GCPs items from the CDR and save as a JSON file.

        :param cog_url: COG URL including COG ID.
        :param output_dir: Output directory.
        :param L1: List Manager to append any failed COG IDs during the pull request process.
        """

        response = self.client.get(cog_url[1], headers = self.headers)

        # If successful response from the server - download the data.
        if response.status_code == 200:
            data        = loads(response.content.decode('utf-8'))

            output_loc = f"{output_dir}/{cog_url[0]}/annotated/gcps"
            DataEng.checkdir(dir_name=output_loc)
            output_file = f"{output_loc}/{cog_url[0]}_gcps.json"
            DataEng.write_json(output_file = output_file, data = data)

        # Otherwise append COG ID information that has failed
        else:
            L1.append([cog_url[0], response.status_code])

"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer.

Date Created: Sept. 2024.

Last Update: Oct. 2024.

About:

Outputs:

Pre-requisites:

Warnings:

Outputs:
"""

# Miscellaneous packages
from tqdm import tqdm
from functools import partial
from multiprocessing import Manager
import httpx
from json import loads
from typing import Union, List
from .gen_cdr import Generic

# Packages used to import custom-made packages outside of relative path.
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Custom-made package
from ..utils import ParallelThread, DataEng


class COG_NGMDB_ID:
    def __init__(self, config: Union[str, dict], fast_api_url="https://api.cdr.land/v1"):

        self.fast_api_url = fast_api_url
        self.client       = httpx.Client(timeout=None)

        kwargs         = DataEng.read_kwargs(config=config)
        token          = kwargs['token']
        cog_ids        = kwargs['cog_ids'] # Must be either a list or a string pointing to a csv file.
        cog_id_field   = kwargs.get('cog_id_field', None) # If cog_ids points to csv file, then state which field it is.

        # Checks if the COG IDs are list or CSV.
        self.cog_ids = Generic.acquire_cog_ids_from_csv(cog_ids      = cog_ids,
                                                        cog_id_field = cog_id_field)

        # Construct authorization headers.
        self.headers = {"accept"        : "application/json",
                        "Authorization" : f"Bearer {token}"}

        # Build COG ID URLs.
        self.cog_urls = self._build_urls()

        # Multithreading to download
        L1 = Manager().list()
        partial_func = partial(self._main_pull, L1=L1)
        ParallelThread(start_method='spawn', partial_func=partial_func, main_list=self.cog_urls)
        self.data_list = L1

    def _build_urls(self) -> List:
        """
        Build COG ID urls to extract COG ID information.

        :return: Return nested list of COG IDs and its respective URL.
        """
        cog_urls = [[cog, f"{self.fast_api_url}/maps/cog/{cog}"]
                    for cog in tqdm(self.cog_ids)]
        return cog_urls

    def _main_pull(self, cog_url, L1):
        """
        Main function to pull annotated legend items from the CDR and save as a JSON file.

        :param cog_url: COG URL including COG ID.
        :param output_dir: Output directory.
        :param L1: List Manager to append any failed COG IDs during the pull request process.
        """

        response = self.client.get(cog_url[1], headers = self.headers)

        if response.status_code == 200:
            data        = loads(response.content.decode('utf-8'))
            L1.append([cog_url[0], data])

        else:
            L1.append([cog_url[0], None])

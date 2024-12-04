"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer

Date Created: August 2024

Date Modified: August 2024

About: Using multithreading to batch pull TIF files from the CDR.

Pre-requisites:
    - Good internet speed (100s Mbps).
    - CSV file containing COG IDs.
    - CDR Token (request admin for one).

Warnings: Must not have security restrictions (e.g., ZScaler) that causes 403 (forbidden) errors.

Outputs:
    - Successful download would be in the following format: "{output_dir}/{cog_id}.tif"
    - Unsuccessful download(s) can be accessed either:
        - "df" variable OR
        - "{output_dir}/failed_cog_ids_tiff.csv"
"""

# CDR API
import cdrc

# Data Engineering related packages
from numpy import array_split
from pandas import DataFrame, concat

# Miscellaneous packages
from tqdm import tqdm
from multiprocessing import cpu_count, Manager
from functools import partial
import time
import random
from typing import Union
from .gen_cdr import Generic

# Importing custom-made packages outside of relative path
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Custom-made packages
from ..utils import ParallelThread, DataEng


class BulkCOG:

    def __init__(self, config: Union[str, dict]):
        """
        :param config: Dictionary or JSON config file that requires the following parameters:
            - token          => Str         => CDR token for access.
            - cog_ids        => List or Str => CSV file containing COG IDs or a list of COG IDs.
            - cog_id_field   => Str         => Name of COG ID Field - only applies if you've listed cog_ids as a csv file.
            - output_dir     => Output directory to store the TIF files being pulled.
        """

        kwargs          = DataEng.read_kwargs(config = config)
        token           = kwargs["token"]
        cog_ids         = kwargs["cog_ids"]
        cog_id_field    = kwargs.get("cog_id_field", None)
        self.output_dir = kwargs["output_dir"]

        # Checks if the COG IDs are list or CSV.
        self.cog_ids = Generic.acquire_cog_ids_from_csv(cog_ids      = cog_ids,
                                                        cog_id_field = cog_id_field)

        # Split the list of COG IDs equally for multithreading across all CPUs.
        self.cogs       = array_split(cog_ids, cpu_count())
        self.client     = cdrc.CDRClient(token = token, output_dir = self.output_dir)
        self.fail_pulls = self._mainPull()

        # If there are failures downloading TIF files from the CDR - export the list of COG IDs.
        if len(self.fail_pulls) > 0:
            failed_cogs = '\n'.join([ff for f in self.fail_pulls for ff in f])
            print(f"COG IDs that have failed:\n{failed_cogs}")
            self.df = concat([DataFrame({"cog_ids" : f}) for f in self.fail_pulls])
            self.df.to_csv(f"{self.output_dir}/failed_cog_ids_tiff.csv", index=False)

    def _download(self, main_list, L1):
        """
        Download TIF files from the COG ID.

        :param main_list: COG ID.

        :param L1: List manager that collects any failed COG IDs.
        """
        try:
            self.client.download_cog(cog_id=main_list)

        except Exception:
            L1.append(main_list)

    def _mainPull(self):
        """
        Multithreading function that downloads several COG IDs at once. The for-loop approach is a way to reduce overwhelming
        server requests.
        """
        fail_pulls = []
        for main_list in tqdm(self.cogs):
            L1 = Manager().list()
            partial_func = partial(self._download, L1=L1)
            ParallelThread(start_method='spawn', partial_func=partial_func, main_list=main_list)
            time.sleep(random.randint(15, 30))

            if len(L1) > 0:
                fail_pulls.append(L1)

        return fail_pulls

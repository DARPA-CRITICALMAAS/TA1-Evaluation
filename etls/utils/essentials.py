"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer

Date Created: 2023

Date Modified: Aug. 2024

About: Three classes for parallelization and data skewness optimization.
    - ParallelPool => Using the imap_unordered method to parallel process across all of your CPUs.
                   => Defaults to number of CPUs available on machine and "spawn" to avoid IPC (interprocess communication).

    - EquitDfs => Specifically catered to DataFrames based on a specific field that compiles data information per row or
                  aggregated form, this class aims to reduce data skewness by allocating a more equitable amount of data
                  per CPU. Equitable is not the same as equal amount. Not all data is same size or equal across processes.

    - ParallelThread => Likewise to ParallelPool, except it uses the threads in the CPUs to conduct I/O (inbound / outbound)
                        operations.
                     => Defaults to number of CPUs available on machine and "spawn".
"""
from multiprocessing import set_start_method, Pool, cpu_count, Process
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from typing import Union
from geopandas import GeoDataFrame
from pandas import DataFrame
from numpy import array, multiply, arange


class ParallelPool:

    def __init__(self, start_method, partial_func, main_list, num_cores = cpu_count()):
        """
        Use the Pool function in multiprocessing to parallel process - equivalent to "launch all in one salvo approach".
        :params start_method: The method to initiate - typically in Linux -> "fork"; Windows -> "spawn".
        :params partial_func: A custom partial function that takes most of the parameters of a custom function to be parallel processed.
        :params main_list: A numpy array list that has been chunked into n number of cores or regular list that will be distributed.
        """

        self._pool(start_method=start_method, partial_func=partial_func, main_list=main_list, num_cores = num_cores)

    def _pool(self, start_method, partial_func, main_list, num_cores):
        """
        Initiate parallel processing.
        :params start_method: The method to initiate - typically in Linux -> "fork"; Windows -> "spawn".
        :params partial_func: A custom partial function that takes most of the parameters of a custom function to be parallel processed.
        :params main_list: A numpy array list that has been chunked into n number of cores.
        """

        set_start_method(method=start_method, force=True)
        with Pool(processes=num_cores) as p:
            max_ = len(main_list)
            with tqdm(total=max_) as pbar:
                for i, _ in enumerate(p.imap_unordered(partial_func, main_list)):
                    pbar.update()
            p.close()
            p.join()


class ParallelProcess:

    def __init__(self, start_method, targeter, parameters, split_val):
        """
        Use the Process function in multiprocessing to parallel process.

        :params start_method: The method to initiate - typically in Linux -> "fork"; Windows -> "spawn".
        :params targeter: The custom function that is to be parallel processed.
        :params parameters: The parameters required in the custom function.
        :params split_val: The list that is to be split into chunks.
        """
        self._process(start_method=start_method,
                      targeter=targeter,
                      parameters=parameters,
                      split_val=split_val)

    def _process(self, start_method, targeter, parameters, split_val):
        """
        Initiate parallel processing.

        :params start_method: The method to initiate - typically in Linux -> "fork"; Windows -> "spawn".
        :params targeter: The custom function that is to be parallel processed.
        :params parameters: The parameters required in the custom function.
        :params split_val: The list that is to be split into chunks.
        """
        set_start_method(method=start_method, force=True)
        processes = []
        for i in range(len(split_val)):
            new_param = (split_val[i],) + parameters[1:]
            p = Process(target=targeter, args=new_param)
            processes.append(p)
            p.start()
        for process in processes:
            process.join()


class EquitDfs:

    def __init__(self, df: Union[DataFrame, GeoDataFrame], geom_length="geom_size"):
        """
        :param df: DataFrame or GeoDataFrame to split into more equitable dataframes.
        :param geom_length: Field name containing information used to facilitate distribution more equitably.
                            Default field name is "geom_size" which has to exist or needs to be calculated for each
                            geometry object. Otherwise, change it to another field.

        Notes: Distribution defaults to the number of CPUs on machine.
        """
        total_sum        = df[geom_length].sum()
        est_distrib      = round(total_sum / cpu_count())
        self.index_split = []

        h, idx = 0, 0
        for j in range(0, cpu_count()):
            i = 0
            try:
                while i < est_distrib:
                    tmp_add = df[geom_length].iloc[h]
                    i = i + tmp_add
                    h += 1

                self.index_split.append(df.iloc[idx:h + 1])
                idx = h + 1

            except IndexError:
                self.index_split.append(df.iloc[idx:])
                break


class RangeBased:

    def __init__(self, df: DataFrame, num_bins=cpu_count()):
        """
        Split the dataframe into equivalent partitions (i.e., equal number of rows as a range-based approach).

        :param df: DataFrame of the dataset.
        :param num_bins: The number of bins (i.e., splits) - defaults to the number of CPUs on machine.
        """

        splitter    = round(len(df) / num_bins)
        pre_consec  = array([splitter] * num_bins)
        range_based = list(multiply(arange(1, num_bins + 1), pre_consec))

        df = df.reset_index()

        multi_dfs = []
        starter   = 0
        for r in range_based:
            multi_dfs.append(df.loc[starter:r])
            starter = r

        self.multi_dfs = multi_dfs


class ParallelThread:

    def __init__(self, start_method, partial_func, main_list):
        """
        Use the ThreadPool function in multiprocessing to parallel thread - equivalent to "launch all in one salvo approach".
        :params start_method: The method to initiate - typically in Linux -> "fork"; Windows -> "spawn".
        :params partial_func: A custom partial function that takes most of the parameters of a custom function to be parallel processed.
        :params main_list: A numpy array list that has been chunked into n number of cores or regular list that will be distributed.
        """
        self._pool(start_method=start_method, partial_func=partial_func, main_list=main_list)

    def _pool(self, start_method, partial_func, main_list):
        """
        Initiate parallel processing.
        :params start_method: The method to initiate - typically in Linux -> "fork"; Windows -> "spawn".
        :params partial_func: A custom partial function that takes most of the parameters of a custom function to be parallel processed.
        :params main_list: A numpy array list that has been chunked into n number of cores.
        """

        set_start_method(method=start_method, force=True)
        with ThreadPool(processes=cpu_count()) as p:
            max_ = len(main_list)
            with tqdm(total=max_) as pbar:
                for i, _ in enumerate(p.imap_unordered(partial_func, main_list)):
                    pbar.update()
            p.close()
            p.join()
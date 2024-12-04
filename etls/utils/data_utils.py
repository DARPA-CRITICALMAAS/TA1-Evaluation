"""
Author: Anastassios Dardas, PhD - Lead Geospatial Computing Engineer

Date Created: 2023

Date Modified: August 2024

About: Two classes for "every day' operations / utilities.
    - DataEng   => For data engineering and miscellaneous functions.
    - NpEncoder => Encode values to serializable JSON.
"""

from json import load, JSONEncoder, dump
from typing import List, Union
import numpy as np
import os
import zipfile

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DataEng:

    def __init__(self):
        pass

    @staticmethod
    def read_json(config_file: str):
        """
        Read a config JSON file - assumes there are no binary conversions required.

        :param config_file: A config JSON file.

        :return: Dictionary that is intended to be keyword arguments.
        """
        with open(config_file, 'r') as f:
            kwargs = load(f)

        return kwargs

    @staticmethod
    def write_json(output_file: str, data: Union[dict, list]):
        """
        Write data to JSON file.

        :param output_file: Path including name of file with .json extension to output.
        :param data: Must be either dictionary or list containing dictionary.
        """
        with open(output_file, "w") as f:
            if isinstance(data, dict):
                dump(data, f, indent=4)

            else:
                dump(data, f)

    @staticmethod
    def flatten(t: List) -> List:
        """
        Flatten a (2D) nested list to a list.

        :param t: Nested List.

        :return: 1D List.
        """
        return [item for sublist in t for item in sublist]

    @staticmethod
    def checkdir(dir_name: str):
        """
        Check if the directory exists and if not, then create it.

        :param dir_name: Intended directory path.
        """
        if os.path.exists(dir_name) is False:
            os.makedirs(dir_name)

    @staticmethod
    def checkfile(file: str):
        """
        Check if the file exists.

        :param file: Name of the file with path included.

        :return: Boolean - True (exists); otherwise False (does not exist).
        """
        if os.path.exists(file) is False:
            return False
        else:
            return True

    @staticmethod
    def filename_fromdir(dir_file: str) -> str:
        """
        Acquire the name of the file by excluding directory path and its file extension.

        :param dir_file: Directory path with the file.

        :return: Name of the file.
        """
        basename = os.path.basename(os.path.splitext(dir_file)[0])
        return basename

    @staticmethod
    def dir_fromfilepath(dir_file: str):
        """
        Get directory path and exclude file name.

        :param dir_file: Directory path with file name.

        :return: Directory path.
        """
        filepath = os.path.dirname(dir_file)
        return filepath

    @staticmethod
    def generic_output_text(output: str, info, method="w"):
        """
        Output information to text file.

        :param output: Path including name of file with .txt extension to output.
        :param info: Data that could be string (ideal), list, dictionary, etc.
        :param method: Defaults to "w" as write/overwrite - could be "wb" to write binary mode or "a" append.
        :return:
        """
        with open(output, method) as data:
            data.write(info)
            data.close()

    @staticmethod
    def last_folder_from_dir(dir_path: str):
        """
        Acquire the last path from the directory path.

        :param dir_path: Directory path.

        :return: Last sub-folder path.
        """
        return os.path.basename(os.path.normpath(dir_path))

    @staticmethod
    def extract_zipfile(zip_file: str, output_dir: str):
        """

        :param zip_file:
        :param output_dir:

        :return:
        """
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        os.remove(zip_file)

    # Check to read keyword arguments from CSV file; otherwise, return as is (i.e., dict).
    @staticmethod
    def read_kwargs(config: str):
        if isinstance(config, str):
            kwargs = DataEng.read_json(config_file=config)
        elif isinstance(config, dict):
            kwargs = config

        return kwargs

    @staticmethod
    def cosine_similarity(data: List):
        vectorizer = TfidfVectorizer()
        X          = vectorizer.fit_transform(data)
        cosine_sim = cosine_similarity(X[0:1], X[1:])
        return cosine_sim

    @staticmethod
    def intersect_2D_np_arrays(array1, array2):
        nrows, ncols = array1.shape
        dtype        = {'names' : ['f{}'.format(i) for i in range(ncols)],
                        'formats' : ncols * [array1.dtype]}

        match_array = np.intersect1d(array1.view(dtype), array2.view(dtype))
        match_array = match_array.view(array1.dtype).reshape(-1, ncols)

        return match_array


class NpEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

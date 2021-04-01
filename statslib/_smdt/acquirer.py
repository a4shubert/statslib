import os

from statslib.utils.file import split_file_path

import gc
import pandas as pd
from datetime import datetime
from statslib.utils.log import Logger
from pathlib import Path
import xlrd

home = str(Path.home())

TEMP_FOLDER = os.path.join(home, 'temp')
if not os.path.exists(TEMP_FOLDER):
    os.mkdir(TEMP_FOLDER)


class Acquirer(object):

    def __init__(self, smart_data=None):
        if smart_data:
            if not hasattr(smart_data, '_is_smart_data'):
                raise TypeError(
                    "smart_data argument must be an instance of a SmartData container."
                )
            self._smart_data = smart_data
        else:
            # we allow for an empty constructor
            pass

        self._Logger = Logger()
        self._logger = self._Logger.get_logger()

    def set_logger(self, file_name, folder, level):
        self._Logger = Logger(file_name, folder, level)
        self._logger = self._Logger.get_logger()

    @classmethod
    def _pandas_from_file(cls, file_path, *args, **kwargs):

        folder, file, extension = split_file_path(file_path)
        _Logger = Logger()
        _logger = _Logger.get_logger()

        if kwargs.get('encoding'):
            encoding = kwargs.pop('encoding')
        else:
            encoding = 'ISO-8859-1'

        result_df = None

        if extension == 'csv':
            chunks = pd.read_csv(
                file_path,
                sep=',',
                chunksize=10**5,
                encoding=encoding,
                *args,
                **kwargs)
            result_df = pd.DataFrame()
            result_df = pd.concat(chunk for chunk in chunks)
            del chunks

        elif extension == 'json':
            result_df = pd.read_json(
                file_path, encoding=encoding, *args, **kwargs)
        elif 'xls' in extension:
            result_df = pd.read_excel(
                file_path,   *args, **kwargs)
        elif 'pickle' in extension or 'pkl' in extension:
            result_df = pd.read_pickle(file_path, *args, **kwargs)
        elif 'txt' in extension:
            result_df = pd.read_table(
                file_path, encoding=encoding, *args, **kwargs)
        elif 'zip' in extension:
            import zipfile
            zp = zipfile.ZipFile(file_path)
            if len(zp.filelist) > 1:
                _logger.debug(
                    "There are more than 1 file in the zip archive. We will concatenate via raw axis, e.g. axis=0. Please"
                    "provide axis=1 argument to concatenate along columns")
                result_df = pd.DataFrame()
                passed_axis = kwargs.get('axis', 0)
                for file_name in zp.filelist:
                    file_name = file_name.filename
                    unzipped_extension = file_name.split('.')[-1]
                    _temp_df = cls._pandas_from_file(
                        zp.open(file_name), extension=unzipped_extension)
                    result_df = pd.concat([result_df, _temp_df],
                                          axis=passed_axis)
                    del _temp_df
                    gc.collect()
        else:
            # raise NotImplementedError(
            #     "The file extension `{}` hasn't been yet implemented!".format(
            #         extension))
            pass

        if isinstance(result_df, pd.Series):
            result_df = result_df.to_frame()

        return result_df

    def from_file(self, file_path, name=None, *args, **kwargs):
        if len(self._smart_data._collections) < 1:
            raise ValueError("SmartData container doesn't have collections.")
        start = datetime.now()
        df = self._pandas_from_file(file_path, *args, **kwargs)
        end = datetime.now()
        self._logger.debug(
            "It took {:g} seconds to read dataFrame from file {}".format(
                (end - start).total_seconds(), os.path.split(file_path)[-1]))

        # AT THIS MOMENT WE NEED TO PUT THE OBTAINED DF TO THE MAIN SMART DATA COLLECTION
        main_collection = getattr(self._smart_data,
                                  self._smart_data._main_collection)
        if name is None:
            folder, file_nm, extension = split_file_path(file_path)
            name = file_nm
        main_collection[name] = df

    def from_folder(self, folder_path, extension=None, **kwargs):
        if len(self._smart_data._collections) < 1:
            raise ValueError("SmartData container doesn't have collections.")
        start = datetime.now()

        files_list = sorted(os.listdir(folder_path))
        if extension is not None:
            files_list = list(filter(lambda fl: extension in fl, files_list))

        file_paths_list = [
            os.path.join(folder_path, file_name) for file_name in files_list
        ]
        file_paths_list = [f for f in file_paths_list if not os.path.isdir(f)]
        res = [
            self.from_file(file_path, **kwargs) for file_path in file_paths_list
        ]

        end = datetime.now()
        self._logger.debug(
            "It took {:.2g} mins to read datasets from folder".format(
                (end - start).total_seconds() / 60))

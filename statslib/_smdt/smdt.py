import sys, os
import copy
import pandas as pd
import datetime
import json
import os
import shutil
import multiprocessing as mp

from multiprocessing.pool import ApplyResult
import itertools

sys.setrecursionlimit(1500)

import threading

from statslib.utils.log import Logger
from statslib.utils.excel import export_obj_to_excel
from statslib.utils.file import create_file_name
from statslib.utils.dates import t_to_str_file_name
from statslib.utils.common import eval_func_tuple
from statslib.utils.common import report_run_time
from statslib.utils.common import flatten_lst

from copyreg import pickle
from types import MethodType


def _get_validated_name(name, is_df=True):
    name = name.replace(' ', '_')
    name = name.replace('-', '_')
    name = name.replace('(', '')
    name = name.replace(')', '')
    name = name.replace('@', '')
    name = name.replace('.', '')

    if is_df:
        if not name.startswith('df_'):
            name = 'df_{}'.format(name)
    else:
        if not name.startswith('obj_'):
            name = 'obj_{}'.format(name)

    return name


class SmartData(object):

    def __init__(self, empty=False):
        self.cond = threading.Condition()
        from statslib._smdt.acquirer import Acquirer

        self._collections = []
        self._main_collection = None
        self._logger = Logger().get_logger()
        self._Logger = Logger()
        self._is_smart_data = True
        self.acquire = Acquirer(smart_data=self)

        if not empty:
            # we add at least one collection: sourced
            self.add_collection('sourced')

    def __add__(self, other):
        pass  # we will be able to combine two collections

    def add_collection(self, name):
        if name not in self._collections:
            _collection = SmartDataCollection(
                smart_data=self, collection_name=name)
            setattr(self, name, _collection)
            self._collections.append(name)
            self._main_collection = name
        else:
            _collection = getattr(self, name)
            _collection.clean_up()
            delattr(self, name)
            self.add_collection(name)

    def _set_logger(self, file_name=None, folder=None, level=None):
        kwargs = dict(
            zip(['file_name', 'folder', 'level'], [file_name, folder, level]))
        kwargs = {key: val for key, val in kwargs.items() if val is not None}

        self._Logger = Logger(**kwargs)
        self._logger = self._Logger.get_logger()

    def __repr__(self):
        CHAR = '='
        if len(self._collections) < 1:
            result = 'SmartData: there are no collections.\n'
            result += CHAR * len(result) + '\n\n'
        else:
            result = '"{}" is a main collection\n'.format(self._main_collection)
            result += CHAR * len(result) + '\n\n'
            result += '\n\n'.join([
                getattr(self, collection_name).__repr__()
                for collection_name in self._collections
            ])
        return result

    def get(self, name):
        # we can form a union of all names in all collection:
        # name = _get_validated_name(name)
        try:
            all_names = {}
            for collection in self._collections:
                collection = getattr(self, collection)
                all_names[collection._collection_name] = collection.names

            _found_collection_df = {
                key: _get_validated_name(name, is_df=True)
                for key, val in all_names.items()
                if _get_validated_name(name, is_df=True) in val
            }
            _found_collection_obj = {
                key: _get_validated_name(name, is_df=False)
                for key, val in all_names.items()
                if _get_validated_name(name, is_df=False) in val
            }

            if len(_found_collection_df.keys()) + len(
                    _found_collection_obj.keys()) > 1:
                raise ValueError("{} found multiple times:\n{}".format(
                    name, 'as df: ' + _found_collection_df.__str__() +
                          '\nas obj: ' + _found_collection_obj.__str__()))
            else:
                if len(_found_collection_df.keys()) > 0:
                    return getattr(
                        self,
                        list(_found_collection_df.keys())[0]).get(name)
                if len(_found_collection_obj.keys()) > 0:
                    return getattr(
                        self,
                        list(_found_collection_obj.keys())[0]).get(name)
        except KeyError:
            raise KeyError("Object {} is not found in SmartData!".format(name))

    @report_run_time(long_format=True, f_name='smart_data_to_folder')
    def to_folder_not_pooled(self, folder=None, ordered=False):
        if folder is None:
            folder = r'C:\temp'
            if not os.path.exists(folder): os.mkdir(folder)
            folder = os.path.join(
                folder, 'SmartData_' + t_to_str_file_name(
                    datetime.datetime.now(), format='%Y_%m_%d_%H_%M_%S'))
            os.mkdir(folder)
        else:
            if '.' in folder:
                raise ValueError("Please provide folder (not file) path")
            if not os.path.isdir(folder):
                os.mkdir(folder)
            else:
                shutil.rmtree(folder)
                os.mkdir(folder)
        self._logger.info('Saving SmartData to {}'.format(folder))

        order_counter = 0
        for collection in self._collections:
            collection = getattr(self, collection)
            if ordered:
                order_counter += 1
            collection.to_folder(folder, order_counter)

    @report_run_time(long_format=True, f_name='smart_data_to_folder')
    def to_folder(self, folder=None, ordered=False):
        if folder is None:
            folder = r'C:\temp'
            folder = os.path.join(
                folder, 'SmartData_' + t_to_str_file_name(
                    datetime.datetime.now(), format='%Y_%m_%d_%H_%M_%S'))
            if not os.path.exists(folder):
                os.mkdir(folder)
        else:
            if '.' in folder:
                raise ValueError("Please provide folder (not file) path")

            if not os.path.isdir(folder):
                os.mkdir(folder)
            else:
                shutil.rmtree(folder)
                os.mkdir(folder)

        Logger().get_logger().info('Saving collection to {}'.format(folder))

        flatten_dict = self._flatten_smart_data(ordered)
        pool = mp.Pool(4)
        res = []
        for collection in list(flatten_dict.keys()):
            dict_of_objects = flatten_dict[collection]
            res.append(
                pool.apply_async(
                    collection_dict_to_folder_pool,
                    args=(folder, collection, dict_of_objects)))
        pool.close()
        pool.join()

        res = [r.get() for r in res]

    def _flatten_smart_data(self, ordered=True):
        """The purpose of this method is to constuct the dicitonary of the form
        {collection_name1 : [object1, object2, object3],
        """
        res_dict = {}
        z = 1
        for collection in self._collections:
            collection = getattr(self, collection)
            res_dict[str(z) + '_' +
                     collection._collection_name if ordered else collection.
                _collection_name] = {
                name: collection.get(name) for name in collection.names
            }
            z += 1
        return res_dict

    def to_excel(self, book=None, mask='SmartData', **kwargs):
        if book is None:
            book = create_file_name(mask=mask, extension='xlsx')
        else:
            if not '.' in book:
                book += '.xlsx'
            if not os.path.exists(book):
                raise FileNotFoundError("Please provide valid path to book!")

        index_df = pd.DataFrame.from_dict({
            collection: getattr(self, collection).names
            for collection in self._collections
        },
            orient='index').T.reindex()

        export_obj_to_excel(index_df, book=book, sheet='summary')

        for collection in self._collections:
            collection = getattr(self, collection)
            collection._function_is_called_from_smartdata = True
            collection.to_excel(book=book, **kwargs)
            collection._function_is_called_from_smartdata = False

        self._logger.info("Saved to: {}".format(book))

    def __setitem__(self, key, value):
        # We just put the datasets into the main collection:
        main_collection = getattr(self, self._main_collection)
        main_collection.__setitem__(key, value)

    def __getitem__(self, item):
        # We'll try to get the item if it's unique
        return self.get(item)

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(self, item)
        except AttributeError as a:
            res = self.get(item)
            if res is None:
                raise a
            else:
                return res

    def apply(self, func_obj=None, name=None, collection=None):
        if func_obj is None:
            raise SyntaxError("Please provide the functional object")
        if name is not None:
            func_obj(self.get(name))
            return

        if collection is not None:
            if collection not in self._collections:
                raise ValueError(
                    "The collection {} isn't found in SmartData collections".
                        format(collection))
            else:
                _tmp_m_collection = self._main_collection
                self._main_collection = collection
                self.apply(func_obj=func_obj, collection=None)
                self._main_collection = _tmp_m_collection
        else:
            main_collection = getattr(self, self._main_collection)
            for name in main_collection.names:
                self._logger.debug("Running {}() for: {}".format(
                    func_obj.__name__, name))
                obj = main_collection.get(name)
                func_obj(obj)

    def apply_make(self, func_obj=None, name=None, **kwargs):
        """
        In this function we'll be making new collection named as f.__name__
        The function here must return dictionary, in such a way we'll be
        adding the value of the dictionary to the key object of smartData collection
        :param func_obj:
        :return:
        """
        if func_obj is None:
            raise SyntaxError("Please provide the functional object")
        collection_name = func_obj.__name__
        self.add_collection(collection_name)
        if name is None:
            # the function doesn't take the obj from the SmartData
            res_dict = func_obj(**kwargs)
        else:
            res_dict = func_obj(self.get(name), **kwargs)
        for key, value in res_dict.items():
            self[key] = value

    def __delattr__(self, name):
        self._collections.remove(name)
        if 'source' not in self._collections:
            self.add_collection('source')
        else:
            getattr(self, 'source').as_main()
        super().__delattr__(name)

    def all_names(self):
        return flatten_lst(getattr(self, col).names for col in self._collections)


class SmartDataCollection(object):

    def __init__(self, smart_data, collection_name):
        self._smart_data = smart_data
        self._collection_name = collection_name
        self._function_is_called_from_smartdata = False

    def __repr__(self):
        df_info = ''
        obj_info = ''
        result = '"{}" collection:'.format(str(self._collection_name))
        if len(self.names) == 0:
            df_info = '\n - there are no elements in the collection'
            result = ''.join([result, df_info])
            return result
        one_ob_description = []
        for name in sorted(self.names):
            my_object = getattr(self, name)
            if isinstance(my_object, pd.Series) or isinstance(
                    my_object, pd.DataFrame):
                rows_count = ' (rows: {:,}'.format(getattr(self, name).shape[0])
                if len(getattr(self, name).shape) > 1:
                    column_count = '; columns: {:,.0f})'.format(
                        getattr(self, name).shape[1])
                else:
                    column_count = '; pd.Series)'

                df_info = '\n      - {}'.format(name + rows_count +
                                                column_count)
                one_ob_description.append(df_info)
            else:
                obj_info = '\n      - {}'.format(name + ' ({})'.format(
                    (type(my_object))))
                one_ob_description.append(obj_info)
        result = result + ''.join(one_ob_description)
        return result

    #
    def as_main(self):
        self._smart_data._main_collection = self._collection_name
        self._smart_data._logger.debug('Collection `{}` set as main.'.format(
            self._collection_name))

    #
    @property
    def names(self):
        return [
            val for val in sorted(dir(self))
            if val.startswith('df_') or val.startswith('obj_')
        ]

    def clean_up(self):
        for name in self.names:
            delattr(self, name)

    def __setitem__(self, name, value):
        if self.names is not None and name in self.names: delattr(self, name)
        if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
            name = _get_validated_name(name, is_df=True)
        else:
            name = _get_validated_name(name, is_df=False)
        setattr(self, name, value)

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(self, item)
        except AttributeError as a:
            res = self.get(item)
            if res is None:
                raise a
            else:
                return res

    def get(self, name):
        if self.names is None:
            raise ValueError("There are no objects in the {} collection".format(
                self._collection_name))
        else:
            if _get_validated_name(name) in self.names:
                return getattr(self, _get_validated_name(name))
            if _get_validated_name(name, is_df=False) in self.names:
                return getattr(self, _get_validated_name(name, is_df=False))

    def to_excel(self, book=None, **kwargs):

        if book is None:
            book = create_file_name(
                mask=self._collection_name, extension='xlsx')
        else:
            if not '.' in book:
                book += '.xlsx'

        for name in self.names:
            print(name)
            Logger().get_logger().info("from collection: {} name: {} ".format(
                self._collection_name, name))
            obj = self.get(name)
            sheet_name = (self._collection_name[:15] + '__' + name.replace(
                'df_', '').replace('obj_', '')[:15])[:30]
            export_obj_to_excel(obj.round(8), book=book, sheet=sheet_name, **kwargs)

    @report_run_time(long_format=True, f_name='smart_data_collection_to_folder')
    def to_folder(self, folder=None, **kwargs):
        if folder is None:
            folder = r'C:\temp'
            folder = os.path.join(
                folder, 'SmartData_' + t_to_str_file_name(
                    datetime.datetime.now(), format='%Y_%m_%d_%H_%M_%S'))
            if not os.path.exists(folder):
                os.mkdir(folder)
            else:
                shutil.rmtree(folder)
                os.mkdir(folder)
            folder = os.path.join(folder, self._collection_name)
            os.mkdir(folder)
        else:
            if '.' in folder:
                raise ValueError("Please provide folder (not file) path")
            if not os.path.isdir(folder):
                os.mkdir(folder)
            else:
                shutil.rmtree(folder)
                os.mkdir(folder)

        Logger().get_logger().info(
            'Saving SmartDataCollection to {}'.format(folder))
        res = []
        pool = mp.Pool()
        for name in self.names:
            obj = self.get(name)
            res.append(
                pool.apply_async(
                    export_obj,
                    args=(obj, name, self._collection_name, folder),
                    kwds=kwargs))
        pool.close()
        pool.join()
        res = [r.get() for r in res]


@report_run_time(long_format=True, f_name='smart_data_collection_to_folder')
def collection_dict_to_folder_pool(collection_folder, collection_name,
                                   dict_of_objects):
    collection_folder = os.path.join(collection_folder, collection_name)
    os.mkdir(collection_folder)
    for obj_name, obj in dict_of_objects.items():
        export_obj(obj, obj_name, collection_name, collection_folder)


def export_obj(obj, name, collection_name, folder=None, **kwargs):
    Logger().get_logger().info('  --from collection: {} name : {}'.format(
        collection_name, name))

    if isinstance(obj, list):
        obj = pd.DataFrame(obj).reset_index()
    if isinstance(obj, pd.Series) or isinstance(obj, pd.DataFrame):
        book = create_file_name(
            mask=name, folder=folder, unique=False, extension='xlsx')
        sheet_name = name.replace('df_', '')
        export_obj_to_excel(obj, book=book, sheet=sheet_name, **kwargs)
    else:
        obj_path = os.path.join(folder, name + '.txt')
        with open(obj_path, 'w') as file:
            file.write(obj.__str__())
    return 0


if __name__ == '__main__':
    pass
    # import pandas as pd
    #
    # myData = SmartData()
    # myData['a'] = 4
    # myData['b'] = pd.DataFrame([1, 2.0123456789123456789], columns=['a'])
    # myData.add_collection('test')
    # myData['k'] = (3, 3)
    # myData.to_folder(ordered=True)

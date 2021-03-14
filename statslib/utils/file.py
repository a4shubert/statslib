import os
import datetime
from statslib.utils.dates import t_to_str_file_name
from statslib.utils.log import Logger
from statslib.utils.common import report_run_time

def create_file_name(mask=None,
                     folder=r'C:\temp',
                     extension='csv',
                     unique=True,
                     silent=True,
                     **kwargs):

    if folder is None:
        folder = os.getcwd()
    if mask is None:
        mask = 'results'
    if unique:
        now_string = t_to_str_file_name(
            datetime.datetime.now(),
            format=kwargs.get('format', '%Y_%m_%d_%H_%M_%S'))
        file_name = '{}_{}.{}'.format(mask, now_string, extension)
    else:
        file_name = '{}.{}'.format(mask, extension)

    file_path = os.path.join(folder, file_name)
    if not silent:
        Logger().get_logger().info('Saved to {}'.format(file_path))
    return file_path


def validate_file_path(file_path):
    if not os.path.exists(file_path):
        raise ValueError("The file path provided {} is not valid.")


def validated_folder_path(folder):
    if not os.path.isdir(folder):
        raise ValueError("The folder provided {} is not valid.".format(folder))


def split_file_path(file_path):
    validate_file_path(file_path)
    folder, file_name = os.path.split(file_path)
    extension = False
    if '.' in file_name:
        file_name, extension = file_name.split('.')

    return folder, file_name, extension


def save_to_json(file_path=None, data=None):
    import json
    with open(file_path, 'w') as f:
        json.dump(data,f)

def read_from_json(file_path=None):
    import json
    with open(file_path, 'r') as f:
        return json.load(f)

@report_run_time(long_format=False)
def zip_folder(archive_name, folder_to_zip):
    import zipfile
    import os, sys
    Logger().get_logger().info('Ziping...')
    zipobj = zipfile.ZipFile(archive_name + '.zip', 'w', zipfile.ZIP_DEFLATED)
    rootlen = len(folder_to_zip) + 1
    for base, dirs, files in os.walk(folder_to_zip):
        for file in files:
            fn = os.path.join(base, file)
            zipobj.write(fn, fn[rootlen:])
    return os.path.abspath(os.path.join(os.getcwd(),archive_name+'.zip'))


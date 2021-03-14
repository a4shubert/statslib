import os
from statslib._pathmap import TEST_FOLDER
from statslib.utils.common import to_namedtuple
from statslib.utils.dframe import to_pd_todatetime
import pandas as pd

_path = os.path.join(TEST_FOLDER, 'data', 'y.csv')
_y_df = pd.read_csv(_path)
_y_df = to_pd_todatetime(_y_df, 'day', day_only=False)
_y_df = _y_df.set_index('day').squeeze()

_path = os.path.join(TEST_FOLDER, 'data', 'x.csv')
_x_df = pd.read_csv(_path)
_x_df = to_pd_todatetime(_x_df, 'day', day_only=False)
_x_df = _x_df.set_index('day').squeeze()
datasets_dict = dict(y=_y_df, X=_x_df)

datasets = to_namedtuple(datasets_dict, False)

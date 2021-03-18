import os

import pandas as pd

from statslib._pathmap import DATA_FOLDER
from statslib.utils.common import to_namedtuple
from statslib.utils.dframe import to_pd_todatetime

_path = os.path.join(DATA_FOLDER, 'oil.csv')
_oil_df = pd.read_csv(_path)
_oil_desc_dict = list(zip(['spirit', 'gravity', 'pressure', 'distil', 'endpoint'], [
    'percentage yield of petroleum spirit',
    "specific gravity of the crude",
    "crude oil vapour pressure, measured in pounds per square inch",
    "the ASTM 10% distillation point, in ◦F",
    "the petroleum fraction end point, in ◦F"
]))

_path = os.path.join(DATA_FOLDER, 'stocks.csv')
_stocks_df = pd.read_csv(_path)
_stocks_df = to_pd_todatetime(_stocks_df, 'day')
_stocks_df.set_index('day', inplace=True)

_path = os.path.join(DATA_FOLDER, 'uschange.csv')
_uschange_df = pd.read_csv(_path)
_uschange_df = to_pd_todatetime(_uschange_df, 'date')
_uschange_df.set_index('date', inplace=True)
_uschange_df.index.freq = 'Q'

datasets_dict = dict(oil={'df': _oil_df, 'desc': _oil_desc_dict},
                     stocks={'df': _stocks_df},
                     uschange={'df': _uschange_df})

datasets = to_namedtuple(datasets_dict, True)

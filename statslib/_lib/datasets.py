import os
from statslib._pathmap import DATA_FOLDER
from statslib.utils.common import to_namedtuple
from statslib.utils.dframe import to_pd_todatetime
import pandas as pd



_path = os.path.join(DATA_FOLDER, 'oil.csv')
_oil_df = pd.read_csv(_path)
_oil_dict = dict(df=_oil_df, desc = list(zip(['spirit', 'gravity', 'pressure', 'distil', 'endpoint'], [
    'percentage yield of petroleum spirit',
    "specific gravity of the crude",
    "crude oil vapour pressure, measured in pounds per square inch",
    "the ASTM 10% distillation point, in ◦F",
    "the petroleum fraction end point, in ◦F"
])))

_path = os.path.join(DATA_FOLDER, 'stocks.csv')
_stocks_df = pd.read_csv(_path)
_stocks_df = to_pd_todatetime(_stocks_df, 'day')
_stocks_df.set_index('day', inplace=True)

datasets_dict = dict(oil=_oil_dict, stocks={'df': _stocks_df})

datasets = to_namedtuple(datasets_dict, True)

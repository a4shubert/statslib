import pandas as pd
from IPython import display


def pretty_columns_names(df):
    df.columns = [
        val.split('[')[0].rstrip() if '[' in val else val
        for val in df.columns.values.tolist()
    ]
    df.columns = [val.lower() for val in df.columns.tolist()]
    df.columns = [val.replace('\n', '') for val in df.columns.tolist()]
    df.columns = [val.replace(' ', '_') for val in df.columns.tolist()]
    df.columns = [val.replace('(', '') for val in df.columns.tolist()]
    df.columns = [val.replace(')', '') for val in df.columns.tolist()]
    df.columns = [val.replace('&', '') for val in df.columns.tolist()]


def get_df_from_excel(file_path, columns=None, *args, **kwargs):
    df = pd.read_excel(file_path, *args, **kwargs)
    if columns is not None:
        for column in columns:
            df[column] = pd.to_datetime(df[column])
    else:
        # looking at the first row to infer datasets types
        for column in df.columns:
            is_str_flag = isinstance(df.loc[0, column], str)
            if is_str_flag:
                try:
                    df[column] = pd.to_datetime(df[column])
                except:
                    pass

    return df


def display_full_df(df):
    if isinstance(df, pd.Series):
        df = df.to_frame()

    pd.set_option('display.max_rows', pd.DataFrame(df).shape[0])
    pd.set_option('display.max_columns', pd.DataFrame(df).shape[1])
    pd.set_option('display.max_colwidth', 1000)
    display.display(pd.DataFrame(df))

    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_colwidth')



import pandas as pd
from IPython import display
import matplotlib.pyplot as plt
import math




def df_see_null_na_values(df):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    from IPython.display import display
    display(df[df.isnull().any(axis=1)])
    display(df[df.isna().any(axis=1)])
    return df[df.isnull().any(axis=1)].index


def to_pd_todatetime(df, col, day_only=False, **kwargs):
    if isinstance(col, list):
        cols = col
        for col in cols:
            df[col] = df[col].apply(lambda t: pd.to_datetime(t, **kwargs))
    else:
        df[col] = df[col].apply(lambda t: pd.to_datetime(t, **kwargs))

    if day_only:
        df[col] = df[col].apply(lambda t: t.date())
    return df


def ddff(df):
    if isinstance(df, pd.Series):
        df = df.to_frame()

    pd.set_option('display.max_rows', pd.DataFrame(df).shape[0])
    pd.set_option('display.max_columns', pd.DataFrame(df).shape[1])
    pd.set_option('display.max_colwidth', 1000)
    display.display(pd.DataFrame(df))

    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_colwidth')


def concat_columnwise(*args):
    if not isinstance(args, list):
        dfs = list(args)
    else:
        dfs = args
    return pd.concat(dfs, axis=1)

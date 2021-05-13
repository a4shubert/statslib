import math
from copy import deepcopy
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import tensorflow as tf

from statslib.utils.common import to_namedtuple


class DesignMatrix:
    def __init__(self, y=None, X=None, f=None, gs=None, add_const=True):
        from statslib._lib.transforms import identical

        y = deepcopy(y)
        X = deepcopy(X)
        gs = deepcopy(gs)
        self.f = None
        self.names = dict()
        self.endog_name = None
        self.exog_names = None
        self._n_y = None
        self._n_x = None
        self.n = None

        if y is None and X is None:
            raise ValueError("Please provide at least one of: y, X!")

        if y is not None:
            if f is None:
                self.f = identical()
            else:
                self.f = deepcopy(f)
            if isinstance(y, pd.DataFrame):
                y = y.squeeze()
            self.endog_name = y.name
            self.y = y
            self.v = self.f(y).rename('v')
            self.names.update({'v': y.name})
            self._n_y = len(self.y)

        if X is not None:
            if gs is None:
                self.gs = [identical()] * len(X.columns)
            else:
                self.gs = gs
            self.exog_names = X.columns.tolist()
            self.X = X
            self._n_x = len(self.X)
            self.names.update(dict(zip([f'g{i}' for i in range(1, len(X.columns) + 1)], X.columns.tolist())))
            if add_const:
                self.names.update({'const': 'const'})
            self._inv_names = {v: k for k, v in self.names.items()}
            if isinstance(gs, dict):
                self.gX = X.agg(gs)
            else:
                self.gX = X.agg(dict(zip(X.columns.tolist(), self.gs)))
            if add_const:
                self.gX['const'] = 1.0
            self.gX.rename(columns=self._inv_names, inplace=True)

            if y is not None:
                self.dm_ext = pd.concat([self.y.rename(self.endog_name), self.v, self.X, self.gX], axis=1)
                self.dm = pd.concat([self.y.rename('y'), self.v, self.gX], axis=1).dropna(axis=0)
                self.gX = self.dm[[name for name in self.names.keys() if name != 'v']]
                self.gX = self.gX[sorted(self.gX.columns)]
            else:
                self.dm_ext = pd.concat([self.X, self.gX], axis=1)
                self.dm = pd.concat([self.gX], axis=1).dropna(axis=0)
                self.gX = self.dm[self.names.keys()]
                self.gX = self.gX[sorted(self.gX.columns)]

        else:
            self.dm_ext = pd.concat([self.y, self.v], axis=1)
            self.dm = self.dm_ext.dropna(axis=0)
            self.gX = None
        self.n = self.dm.shape[0]
        self.dm_ext.index.name = 't'
        self.dm.index.name = 't'
        self.names_tpl = to_namedtuple(self.names, True)

    def describe(self, figsize=(8 * 1.6, 8), g_form=False):
        if g_form:
            endog_name = self.x_to_g(self.endog_name)
            exog_names = self.x_to_g(self.exog_names)
        else:
            endog_name = self.endog_name
            exog_names = self.exog_names
        if self.exog_names:
            if self.endog_name:
                lst = [self.dm_ext[endog_name].describe()] + [self.dm_ext[c].describe() for c in exog_names]
            else:
                lst = [self.dm_ext[c].describe() for c in exog_names]
        else:
            lst = [self.dm_ext[endog_name].describe()]
        res_df = pd.concat(lst, axis=1)
        if res_df.T.shape[0] == 1:
            fig, ax = plt.subplots(figsize=figsize)
            res_df.drop('count', axis=0).T.plot(kind='bar', ax=ax)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            _ = self.dm_ext[[endog_name]+exog_names].melt(var_name=' ', value_name='')
            sns.violinplot(x=' ', y='', data=_, ax=ax)
            # plt.xticks(range(0, len(_.index)), _.index)
            # ax.tick_params(axis='x', rotation=45)
        return res_df

    def seasonal_decompose(self, **kwargs):
        from statslib._lib.explore import decompose_seasonal_stl
        results = decompose_seasonal_stl(self.y, **kwargs)
        return results

    def plot_scatter_lowess(self, lowess_dict=None, drop_names=None, g_form=False):
        if lowess_dict is None:
            lowess_dict = dict()
        if drop_names is None:
            drop_names = list()
        exog_names = [k for k in self.exog_names if k not in drop_names]
        combinations = list(product([self.endog_name], exog_names))
        L = 2
        K = math.ceil(len(combinations) / L)
        i = j = 0
        if K == 1:
            K += 1
        mask = [self.endog_name] + exog_names
        if g_form:
            df = self.dm_ext[self.x_to_g(mask)]
        else:
            df = self.dm_ext[mask]
        fig, axs = plt.subplots(K, L, figsize=(15, 15))
        for combination in combinations:
            combination = list(combination)
            if g_form:
                combination = self.x_to_g(combination)
            df_combo = df[combination]
            x = df_combo.iloc[:, 1].values
            y = df_combo.iloc[:, 0].values
            axs[i, j].scatter(x, y, color='#8EBA42')
            pd.DataFrame(sm.nonparametric.lowess(y, x, **lowess_dict), columns=['x', 'y']).set_index('x').plot(
                ax=axs[i, j])
            axs[i, j].legend(['lowess'])
            axs[i, j].set_ylabel(combination[0])
            axs[i, j].set_xlabel(self.g_to_x(combination[1]))
            j += 1
            if j % L == 0:
                i += 1
                j = 0
        plt.tight_layout()
        plt.show()

    def plot(self, only_names=None, drop_names=None, g_form=False, **kwargs):
        if drop_names is None:
            drop_names = list()
        else:
            drop_names = self.g_to_x(drop_names)
        if only_names is None:
            only_names = self.names.values()
        else:
            drop_names = list()
            only_names = self.g_to_x(only_names)
        from statslib.utils.common import flatten_lst
        from statslib.utils.plots import plot_to_grid
        if g_form:
            mask = flatten_lst(
                [[v] + [k] for k, v in self.names.items() if k != 'const' and v not in drop_names and v in only_names])
        else:
            mask = flatten_lst(
                [[v] for k, v in self.names.items() if k != 'const' and v not in drop_names and v in only_names])
        plot_to_grid(self.dm_ext[mask], plots_per_row=2, title='', **kwargs)

    def plot_covariate_vs_lag(self, covariate_name, up_to_lag, g_form=False):
        if g_form:
            covariate_name = self.x_to_g(covariate_name)
        h = up_to_lag
        cov_df = self.dm_ext[covariate_name].dropna()
        lagged_df = pd.concat([cov_df] + [cov_df.shift(i).rename(f'Lag{i}_{covariate_name}') for i in range(1, h + 1)],
                              axis=1).dropna()
        y_lagged = lagged_df[covariate_name]
        X_lagged = lagged_df.drop([covariate_name], axis=1)
        DM_lagged = DesignMatrix(y_lagged, X=X_lagged)
        DM_lagged.plot_scatter_lowess(lowess_dict=dict())

    def plot_dependent_vs_covariage_lag(self, covariate_name, up_to_lag, g_form=False):
        if g_form:
            covariate_name = self.x_to_g(covariate_name)

        h = up_to_lag
        cov_df = self.dm_ext[covariate_name].dropna()
        lagged_df = pd.concat([cov_df] + [cov_df.shift(i).rename(f'Lag{i}_{covariate_name}') for i in range(1, h + 1)],
                              axis=1).dropna()

        y_lagged = self.y.rename(self.endog_name)
        X_lagged = lagged_df.drop([covariate_name], axis=1)
        DM_lagged = DesignMatrix(y_lagged, X=X_lagged, f=self.f)

        DM_lagged.plot_scatter_lowess(lowess_dict=dict(), g_form=g_form)

    def g_to_x(self, l):
        if not isinstance(l, list):
            l = [l]
        if len(set([c for c in l if c!= 'const']).intersection((self.names.values()))) >= 1:
            return l if len(l) > 1 else l[0]
        res = list(map(self.names.get, l))
        if len(res) == 1:
            return res[0]
        else:
            return res

    def x_to_g(self, l):
        if not isinstance(l, list):
            l = [l]
        if len(set([c for c in l if c!= 'const']).intersection(self._inv_names.values())) >= 1:
            return l if len(l) > 1 else l[0]
        res = list(map(self._inv_names.get, l))
        if len(res) == 1:
            return res[0]
        else:
            return res


class WindowGenerator:
    def __init__(self, input_width, label_width, shift, train_df, val_df, test_df, label_columns=None):
        # Store the raw dat
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}

        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self.config_dict = None

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in
                 self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def config(self, targets=None, sequence_stride=1, shuffle=True, batch_size=32, **kwargs):
        self.config_dict = dict(
            targets=targets,
            sequence_stride=sequence_stride,
            shuffle=shuffle,
            batch_size=batch_size
        )
        if kwargs is not None:
            self.config_dict.update(kwargs)

    def make_dataset(self, data: pd.DataFrame, ):
        if self.config_dict is None:
            raise ValueError('Please run .config() method!')
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=self.config_dict['targets'],
            sequence_length=self.total_window_size,
            sequence_stride=self.config_dict['sequence_stride'],
            shuffle=self.config_dict['shuffle'],
            batch_size=self.config_dict['batch_size'],
        )
        ds = ds.map(self.split_window)
        return ds

    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result

    def plot(self, plot_col=None, model=None, max_subplots=3, feedback=False):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]_batch_{n}')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)

            from statslib.utils.common import flatten_lst as fl
            if model is not None:
                predictions = model.predict(inputs)

                if len(self.example) < 1:
                    predictions = fl(predictions)

                    plt.scatter(self.label_indices, predictions,
                                marker='X', edgecolors='k', label='Predictions',
                                c='#ff7f0e', s=64)
                else:
                    if feedback:
                        predictions = tf.transpose(predictions, [0, 2, 1])
                        predictions = predictions[n, 0, :]

                    else:
                        predictions = predictions[n, :]
                    plt.scatter(self.label_indices, predictions,
                                marker='X', edgecolors='k', label='Predictions',
                                c='#ff7f0e', s=64)

            # if n == 0:
            #     plt.legend()

            plt.xlabel('Time')

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

import math
from copy import deepcopy
from itertools import product

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

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
                self.gX = sm.tools.tools.add_constant(self.gX)
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
        if self._n_x is not None and self._n_y is not None and self._n_x != self._n_y:
            print('WARNING: y and X dimensions are not the same!')
            self.n = self.dm.shape[0]
        else:
            self.n = self._n_y if self._n_y is not None else self._n_x
        self.dm_ext.index.name = 't'
        self.dm.index.name = 't'
        self.names_tpl = to_namedtuple(self.names, True)

    def describe(self, figsize=(8 * 1.6, 8)):
        if self.exog_names:
            if self.endog_name:
                lst = [self.dm_ext[self.endog_name].describe()] + [self.dm_ext[c].describe() for c in self.exog_names]
            else:
                lst = [self.dm_ext[c].describe() for c in self.exog_names]
        else:
            lst = [self.dm_ext[self.endog_name].describe()]
        res_df = pd.concat(lst, axis=1)
        if res_df.T.shape[0] == 1:
            fig, ax = plt.subplots(figsize=figsize)
            res_df.drop('count', axis=0).T.plot(kind='bar', ax=ax)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            plot_df = res_df.drop('count', axis=0).T.reindex()
            plot_df.plot(ax=ax)
            plt.xticks(range(0, len(plot_df.index)), plot_df.index)
            ax.tick_params(axis='x', rotation=45)
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
            axs[i, j].set_xlabel(combination[1])
            j += 1
            if j % L == 0:
                i += 1
                j = 0
        plt.tight_layout()
        plt.show()

    def plot(self, only_names=None, drop_names=None, **kwargs):
        if drop_names is None:
            drop_names = list()
        else:
            drop_names = self.g_to_x(drop_names)
        if only_names is None:
            only_names = self.names.values()
        else:
            only_names = self.g_to_x(only_names)
        from statslib.utils.common import flatten_lst
        from statslib.utils.plots import plot_to_grid
        mask = flatten_lst(
            [[v, k] for k, v in self.names.items() if k != 'const' and v not in drop_names and v in only_names])
        plot_to_grid(self.dm_ext[mask], plots_per_row=2, title='', **kwargs)

    def plot_covariate_vs_lag(self, covariate_name, up_to_lag):
        h = up_to_lag
        cov_df = self.dm_ext[covariate_name].dropna()
        lagged_df = pd.concat([cov_df] + [cov_df.shift(i).rename(f'Lag{i}_{covariate_name}') for i in range(1, h + 1)],
                              axis=1).dropna()
        y_lagged = lagged_df[covariate_name]
        X_lagged = lagged_df.drop([covariate_name], axis=1)
        DM_lagged = DesignMatrix(y_lagged, X=X_lagged)
        DM_lagged.plot_scatter_lowess(lowess_dict=dict())

    def plot_dependent_vs_covariage_lag(self, covariate_name, up_to_lag):
        h = up_to_lag
        cov_df = self.dm_ext[covariate_name].dropna()
        lagged_df = pd.concat([cov_df] + [cov_df.shift(i).rename(f'Lag{i}_{covariate_name}') for i in range(1, h + 1)],
                              axis=1).dropna()
        y_lagged = self.y.rename(self.endog_name)
        X_lagged = lagged_df.drop([covariate_name], axis=1)
        DM_lagged = DesignMatrix(y_lagged, X=X_lagged)
        DM_lagged.plot_scatter_lowess(lowess_dict=dict())

    def g_to_x(self, l):
        if len(set(l).intersection((self.names.values()))) > 1:
            return l
        else:
            return list(map(self.names.get, l))

    def x_to_g(self, l):
        if len(set(l).intersection(self._inv_names.values())) > 1:
            return l
        return list(map(self._inv_names.get, l))

import pandas as pd
import statsmodels.api as sm
import math
from itertools import product
import matplotlib.pyplot as plt

from IPython.display import display


class DesignMatrix:
    def __init__(self, y=None, X=None, f=None, gs=None, add_const=True):

        self.f = None
        self.names = dict()
        self.exog_name = None
        self.endog_names = None
        self._n_y = None
        self._n_x = None
        self.n = None

        if y is None and X is None:
            raise ValueError("Please provide at least one of: y, X!")

        if y is not None:
            if f is None:
                from statslib._lib.transforms import identical
                self.f = identical()
            else:
                self.f = f
            if isinstance(y, pd.DataFrame):
                y = y.squeeze()
            self.exog_name = y.name
            self.y = y
            self.v = self.f(y).rename('v')
            self.names.update({'v': y.name})
            self._n_y = len(self.y)

        if X is not None:
            if gs is None:
                self.gs = [lambda s: s] * len(X.columns)
            else:
                self.gs = gs
            self.endog_names = X.columns.tolist()
            self.X = X
            self._n_x = len(self.X)
            self.names.update(dict(zip([f'g{i}' for i in range(1, len(X.columns) + 1)], X.columns.tolist())))
            if add_const:
                self.names.update({'const': 'const'})
            self._inv_names = {v: k for k, v in self.names.items()}
            self.gX = X.agg(dict(zip(X.columns.tolist(), self.gs)))
            if add_const:
                self.gX = sm.tools.tools.add_constant(self.gX)
            self.gX.rename(columns=self._inv_names, inplace=True)

            if y is not None:
                self.dm_ext = pd.concat([self.y.rename(self.exog_name), self.v, self.X, self.gX], axis=1)
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

    def describe(self, figsize=(8 * 1.6, 8)):
        if self.endog_names:
            lst = [self.dm_ext[self.exog_name].describe()] + [self.dm_ext[c].describe() for c in self.endog_names]
        else:
            lst = [self.dm_ext[self.exog_name].describe()]

        res_df = pd.concat(lst, axis=1)
        if res_df.T.shape[0] == 1:
            res_df.drop('count', axis=0).T.plot(figsize=figsize, kind='bar')
        else:
            res_df.drop('count', axis=0).T.plot(figsize=figsize)
        return res_df

    def seasonal_decompose(self, **kwargs):
        from statslib._lib.explore import decompose_seasonal_stl
        results = decompose_seasonal_stl(self.y, **kwargs)
        return results

    def plot_scatter_lowess(self, lowess_dict=None):
        if lowess_dict is None:
            lowess_dict = dict()
        combinations = list(product([self.exog_name], self.endog_names))
        L = 2
        K = math.ceil(len(combinations) / L)
        i = j = 0
        if K == 1:
            K += 1
        mask = [self.exog_name] + self.endog_names
        df = self.dm_ext[mask]
        fig, axs = plt.subplots(K, L, figsize=(15, 15))
        for combination in combinations:
            combination = list(combination)
            df_combo = df[combination]
            x = df_combo.iloc[:, 0].values
            y = df_combo.iloc[:, 1].values
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

    def plot(self):
        from statslib.utils.common import flatten_lst
        from statslib.utils.plots import plot_to_grid
        mask = flatten_lst([[v, k] for k, v in self.names.items() if k != 'const'])
        plot_to_grid(self.dm_ext[mask], plots_per_row=2, title='Design Matrix')

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
        y_lagged = self.y.rename(self.exog_name)
        X_lagged = lagged_df.drop([covariate_name], axis=1)
        DM_lagged = DesignMatrix(y_lagged, X=X_lagged)
        DM_lagged.plot_scatter_lowess(lowess_dict=dict())

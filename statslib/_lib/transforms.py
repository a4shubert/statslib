from abc import ABCMeta as _ABCMeta
from abc import abstractmethod as _abstractmethod

import numpy as _np
import pandas as _pd
import pandas as pd
import statsmodels.api as _sm


class _GeneralTransform(metaclass=_ABCMeta):
    @_abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @_abstractmethod
    def inv(self, v, y0, idx):
        pass

    @classmethod
    def validate_inv_input(cls, v, y0, idx):
        if y0 is not None:
            if not v.isnull().any():
                v = pd.concat([y0, v], axis=0).sort_index()
                idx = v.index
                v = v.values.tolist()
            if isinstance(y0, _pd.Series) or isinstance(y0, _pd.DataFrame):
                y0 = y0.values.tolist()
        return v, y0, idx


class pct_change(_GeneralTransform):
    """
    Percentage change:
    f(y_t) = \frac{y_t - y_{t-n}} / {y_{t-n}}
    """

    def __init__(self, n=1):
        self.n = n

    def __call__(self, y, **kwargs):
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()
        self.y0 = y.iloc[:self.n].values.tolist()
        self.idx = y.index
        v = y.pct_change(self.n)
        return v.rename('v')

    def inv(self, v, y0=None, idx: list = None):
        vidx = idx
        v, y0, idx = self.validate_inv_input(v, y0, idx)
        idx = self.idx if idx is None else idx
        z_vals = self.y0 if y0 is None else y0
        v_vals = v[self.n:]
        v_vals = [1 + k for k in v_vals]
        z_vals.extend(v_vals)
        y_hat = _pd.concat([_pd.Series(_np.cumprod(z_vals[self.n - i:][::self.n]), index=idx[self.n - i:][::self.n])
                            for i in list(range(self.n, 0, -1))], axis=0).sort_index().rename('y_hat')

        return y_hat.loc[vidx] if vidx is not None else y_hat


class log_return(_GeneralTransform):
    """
    Percentage change:
    f(y_t) = \log\left(\frac{y_t}{y_{t-n}}\right)
    """

    def __init__(self, n):
        self.n = n

    def __call__(self, y, **kwargs):
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()
        self.y0 = y.iloc[:self.n].values.tolist()
        self.idx = y.index
        v = _np.log(y.divide(y.shift(self.n)))
        return v.rename('v')

    def inv(self, v, y0: list = None, idx: list = None):
        vidx = idx
        v, y0, idx = self.validate_inv_input(v, y0, idx)
        idx = self.idx if idx is None else idx
        z_vals = self.y0 if y0 is None else y0
        v_vals = v[self.n:]
        v_vals = [_np.exp(k) for k in v_vals]
        z_vals.extend(v_vals)
        y_hat = _pd.concat([_pd.Series(_np.cumprod(z_vals[self.n - i:][::self.n]), index=idx[self.n - i:][::self.n])
                            for i in list(range(self.n, 0, -1))], axis=0).sort_index().rename('y_hat')
        return y_hat.loc[vidx] if vidx is not None else y_hat


class difference_operator(_GeneralTransform):
    """
    Time series difference operator:
    f(y_t) = \nabla^d\nabla_s^D(y_t) = (1-L)^d (1-L_s)^D y_t =  v_t
    """

    def __init__(self, k_diffs, k_seasonal_diffs=None, seasonal_periods=1, inv_specification=None):
        if inv_specification is None:
            inv_specification = [[], []]
        self.k_diffs = k_diffs
        self.k_seasonal_diffs = k_seasonal_diffs
        self.seasonal_periods = 0 if k_seasonal_diffs is None else seasonal_periods
        if k_seasonal_diffs is not None and seasonal_periods == 0:
            raise ValueError('Please provide seasonal_period!')
        self.n = self.k_diffs + self.seasonal_periods
        self.inv_specification = inv_specification

    def __call__(self, y, **kwargs):
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()
        self.y0 = y.iloc[: self.n].values.tolist()
        self.idx = y.index
        v = _sm.tsa.statespace.tools.diff(y, self.k_diffs, self.k_seasonal_diffs, self.seasonal_periods)
        v = _pd.concat([y, v], axis=1).iloc[:, 1]
        return v.rename('v')

    def inv(self, v, y0=None, idx=None):
        vidx = idx
        v, y0, idx = self.validate_inv_input(v, y0, idx)
        idx = self.idx if idx is None else idx
        z_vals = self.y0 if y0 is None else y0
        v_vals = v[self.n:]
        z_vals.extend(v_vals)
        counter = 0
        while (self.n + counter) < len(z_vals):
            v = z_vals[self.n + counter]
            p = z_vals[counter:][:self.n]
            coefs, elems = self.inv_specification
            running_number = 0
            for i in range(len(coefs)):
                running_number += coefs[i] * p[self.n - elems[i]]
            y_hat = v - running_number
            z_vals[self.n + counter] = y_hat
            counter += 1
        y_hat = _pd.Series(z_vals, index=idx).rename('y_hat')
        return y_hat.loc[vidx] if vidx is not None else y_hat


class min_max(_GeneralTransform):
    """
    v_t = \dfrac{y-y.min()}{y.max()-y.min()}
    """
    def __init__(self, n=None):
        self.n = n

    def __call__(self, y, *args, **kwargs):
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()
        self.y0 = y.iloc[:self.n].values.tolist()
        self.idx = y.index
        v = (y - y.min()) / (y.max() - y.min())
        return v.rename('v')

    def inv(self, v, y0, idx):
        raise NotImplementedError()




class standardize(_GeneralTransform):
    """
    v_t = \dfrac{y_t-\bar{y}_{t-n..t}}{\sigma_{t-n..t}}
    """

    def __init__(self, n=None):
        self.n = n

    def __call__(self, y, *args, **kwargs):
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()
        self.y0 = y.iloc[:self.n].values.tolist()
        self.idx = y.index
        v = (y - y.mean()) / y.std()
        return v.rename('v')

    def inv(self, v, y0, idx):
        raise NotImplementedError()


class identical(_GeneralTransform):
    """
    v_t = y_t
    """
    def __init__(self):
        self.n = 1

    def __call__(self, y, *args, **kwargs):
        if isinstance(y, pd.DataFrame):
            y = y.squeeze()
        return y.rename('v')

    def inv(self, v, y0, idx):
        return v

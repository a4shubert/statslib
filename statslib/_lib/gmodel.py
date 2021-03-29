import inspect
import math as _math
from copy import deepcopy

import matplotlib.pyplot as _plt
import numpy as np
import pandas as pd
import statsmodels.api as _sm

from statslib._lib.gcalib import CalibType


class GeneralModel:
    def __init__(self, gc, DM):
        self.gc = deepcopy(gc)
        self.DM = deepcopy(DM)
        self.calibrator = None
        self.fitted = None
        self.v_hat = None
        self.y0 = None
        self.y_hat = None
        self.residuals = None

    def exog(self, idx):
        return self.DM.gX.iloc[idx] if self.DM.gX is not None else None

    def endog(self, idx):
        return self.DM.dm.v.iloc[idx]

    def fit(self, idx, **kwargs):

        if self.gc.calib_type is CalibType.sm:
            self.calibrator = self.gc.cf(endog=self.endog(idx),
                                         exog=self.exog(idx),
                                         **self.gc.kwargs)

            self.fitted = self.calibrator.fit(**kwargs)

        if self.gc.calib_type is CalibType.sk:
            self.calibrator = self.gc.cf(**self.gc.kwargs)
            self.fitted = self.calibrator.fit(self.exog(idx), self.endog(idx))

        self.y0 = self.DM.dm.y.iloc[idx].tail(self.DM.f.n)

    def forecast(self, idx):
        def sumofsq(x, axis=0):
            """Helper function to calculate sum of squares along first axis"""
            return np.sum(x ** 2, axis=axis)

        self.forecast_index = idx

        if 'start' in inspect.signature(self.fitted.predict).parameters:
            self.v_hat = self.fitted.predict(
                self.endog(idx).index.min(),
                self.endog(idx).index.max(),
                exog=self.exog(idx))
        else:
            if self.gc.calib_type is CalibType.sm:
                self.v_hat = self.fitted.predict(exog=self.exog(idx))
            if self.gc.calib_type is CalibType.sk:
                self.v_hat = self.fitted.predict(self.exog(idx))
                self.v_hat = pd.Series(self.v_hat, index=self.exog(idx).index).rename('v_hat')
        self.y_hat = self.DM.f.inv(self.v_hat, y0=self.y0, idx=self.v_hat.index)
        try:
            self.residuals = self.DM.dm.loc[self.v_hat.index]['v'].values - self.v_hat.values
            sigma2 = 1.0 / self.fitted.nobs * sumofsq(self.residuals)
            self.std_residuals = self.residuals / np.sqrt(sigma2)

            self.residuals = pd.Series(self.std_residuals, index=self.v_hat.index)
            self.std_residuals = pd.Series(self.std_residuals, index=self.v_hat.index)

        except Exception:
            pass

    def plot_diagnostics(self, figsize=(15, 15), drop_names=None):
        if drop_names is None:
            drop_names = list()
        std_resid = self.std_residuals
        if std_resid is not None:
            fig, axs = _plt.subplots(3, 2, figsize=figsize)
            from statslib.utils.plots import get_standard_colors
            clrs = get_standard_colors()
            std_resid.plot(ax=axs[0, 0], color=clrs[1])
            axs[0, 0].hlines(0, self.v_hat.index.min(), self.v_hat.index.max())
            axs[0, 0].set_title('Standardized residuals')

            axs[0, 1].hist(std_resid.values, density=True)
            from scipy.stats import gaussian_kde, norm
            kde = gaussian_kde(std_resid)
            xlim = (-1.96 * 2, 1.96 * 2)
            x = np.linspace(xlim[0], xlim[1])
            axs[0, 1].plot(x, kde(x), label="KernelDensityEstimator")
            axs[0, 1].plot(x, norm.pdf(x), label="N(0,1)")
            axs[0, 1].set_xlim(xlim)
            axs[0, 1].legend()
            axs[0, 1].set_title("Histogram plus estimated density")

            _sm.graphics.qqplot(std_resid.values, line='q', fit=True, ax=axs[1, 0])
            axs[1, 0].set_title('Normal QQ Plot')

            _sm.graphics.tsa.plot_acf(std_resid, ax=axs[1, 1])
            axs[1, 1].set_title('Correlogram')

            axs[2, 0].scatter(self.fitted.fittedvalues, self.residuals.values)
            axs[2, 0].hlines(0, min(self.fitted.fittedvalues), max(self.fitted.fittedvalues))
            axs[2, 0].set_xlabel('fitted')
            axs[2, 0].set_ylabel('resid')
            axs[2, 0].set_title('Fitted values vs. Residuals')

            axs[2, 1].scatter(range(len(self.std_residuals)), self.std_residuals.values)
            axs[2, 1].hlines(0, 0, len(self.std_residuals.values))
            axs[2, 1].set_xlabel('index')
            axs[2, 1].set_ylabel('std_resid')
            axs[2, 1].set_title('Index plot of standardized residuals')

            _plt.tight_layout()
            _plt.show()
            print(" ")
            L = 2
            K = _math.ceil(len([k for k in self.DM.exog_names if k not in drop_names]) / L)
            i = j = 0
            fig, axs = _plt.subplots(K, L, figsize=(15, 15))
            for curve in self.DM.exog_names:
                if curve not in drop_names:
                    x_vals = self.DM.dm_ext[curve].iloc[self.forecast_index].values.tolist()

                    axs[i, j].scatter(x_vals, self.std_residuals.values)
                    axs[i, j].hlines(0, min(x_vals), max(x_vals))
                    axs[i, j].set_xlabel(curve)
                    axs[i, j].set_ylabel('std_res')
                    j += 1
                    if j % L == 0:
                        i += 1
                        j = 0
            _plt.suptitle('Standardized Residuals vs. Explanatory Variable')
            _plt.tight_layout(pad=3)
            _plt.show()

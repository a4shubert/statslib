import inspect
from statslib._lib.gcalib import CalibType
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


class GeneralModel:
    def __init__(self, gc, DM):
        self.gc = gc
        self.DM = DM
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
            self.y0 = self.DM.dm.y.iloc[idx].tail(self.DM.f.n)

    def forecast(self, idx):
        def sumofsq(x, axis=0):
            """Helper function to calculate sum of squares along first axis"""
            return np.sum(x ** 2, axis=axis)

        if 'start' in inspect.signature(self.fitted.predict).parameters:
            self.v_hat = self.fitted.predict(
                self.endog(idx).index.min(),
                self.endog(idx).index.max(),
                exog=self.exog(idx))
        else:
            self.v_hat = self.fitted.predict(exog=self.exog(idx))
        self.y_hat = self.DM.f.inv(self.v_hat, y0=self.y0, idx=self.v_hat.index)
        try:
            self.residuals = self.DM.dm.loc[self.v_hat.index]['v'].values - self.v_hat.values
            sigma2 = 1.0 / self.fitted.nobs * sumofsq(self.residuals)
            self.std_residuals = self.residuals / np.sqrt(sigma2)

        except IndexError:
            pass

    def plot_diagnostics(self, figsize=(9 * 1.6, 9)):
        std_resid = self.std_residuals
        if std_resid is not None:
            fig, axs = plt.subplots(2, 2, figsize=figsize)
            from statslib.utils.plots import get_standard_colors
            clrs = get_standard_colors()
            pd.Series(std_resid, index=self.v_hat.index).plot(ax=axs[0, 0], color=clrs[1])
            axs[0, 0].hlines(0, self.v_hat.index.min(), self.v_hat.index.max())
            axs[0, 0].set_title('Standardized residuals')

            axs[0, 1].hist(std_resid, density=True)
            from scipy.stats import gaussian_kde, norm
            kde = gaussian_kde(std_resid)
            xlim = (-1.96 * 2, 1.96 * 2)
            x = np.linspace(xlim[0], xlim[1])
            axs[0, 1].plot(x, kde(x), label="KernelDensityEstimator")
            axs[0, 1].plot(x, norm.pdf(x), label="N(0,1)")
            axs[0, 1].set_xlim(xlim)
            axs[0, 1].legend()
            axs[0, 1].set_title("Histogram plus estimated density")

            sm.graphics.qqplot(std_resid, line='q', fit=True, ax=axs[1, 0])
            axs[1, 0].set_title('Normal QQ Plot')

            sm.graphics.tsa.plot_acf(std_resid, ax=axs[1, 1])
            axs[1, 1].set_title('Correlogram')

            plt.tight_layout()
            plt.show()

import matplotlib.pyplot as _plt
import statsmodels.api as _sm

def plot_qq_plot(s, figsize=(7, 9), **kwargs):
    fig, ax = _plt.subplots(figsize=figsize)
    _sm.graphics.qqplot(s, line='q', fit=True, ax=ax, **kwargs)
    ax.set_title('Normal QQ Plot')
    _plt.tight_layout()
    _plt.show()


def plot_acf_pcf(s, figsize=(9 * 1.6, 9), kwargs_acf={}, kwargs_pacf={}):
    fig = _plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(211)
    fig = _sm.graphics.tsa.plot_acf(s, lags=kwargs_acf.pop('lags', 30), ax=ax1, **kwargs_acf)
    ax1.set_title('Correlogram')
    ax2 = fig.add_subplot(212)
    fig = _sm.graphics.tsa.plot_pacf(s, lags=kwargs_pacf.pop('lags', 30), ax=ax2, **kwargs_pacf)
    _plt.show()

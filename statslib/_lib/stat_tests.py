import pandas as _pd
import statsmodels.api as sm
from statsmodels.compat import lzip
from statsmodels.stats.stattools import jarque_bera as _jarque_bera
from statsmodels.tsa.stattools import adfuller as _adfuller
from statsmodels.tsa.stattools import kpss as _kpss


def test_jarque_bera(s, alpha=0.01, **kwargs):
    print('H0: sample datasets have the skewness and kurtosis matching a normal distribution', end='\n\n')
    print('Results of Jarque Bera Test:')
    jqberatest = _jarque_bera(s, **kwargs)
    jq_output = _pd.Series(jqberatest, index=['Test Statistic', 'p-value', 'skew', 'kurtosis'])
    print(jq_output, end='\n\n')
    pvalue = jqberatest[1]
    if pvalue < alpha:
        print(f'p-value {pvalue:.4f} is less alpha {alpha} => Reject H0')
    else:
        print('Can NOT reject H0')


def test_kpss(timeseries, alpha=0.01, **kwargs):
    trend_name = kwargs.get('regression', 'constant')
    if trend_name == 'ct':
        trend_name = 'trend'
    print('H0: observable time series is stationary around a {}'.format(trend_name), end='\n\n')
    print('Results of KPSS Test:')
    kpsstest = _kpss(timeseries, **kwargs)
    kpss_output = _pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output, end='\n\n')
    pvalue = kpsstest[1]
    if pvalue < alpha:
        print(f'p-value {pvalue:.4f} is less alpha {alpha} => Reject H0')
    else:
        print('Can NOT reject H0')


def test_adf(timeseries, alpha=0.01, **kwargs):
    print('H0: unit root present in the time series', end='\n\n')
    print('Results of Dickey-Fuller Test:')
    dftest = _adfuller(timeseries, **kwargs)
    dfoutput = _pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput, end='\n\n')
    pvalue = dftest[1]
    if pvalue < alpha:
        print(f'p-value {pvalue:.4f} is less alpha {alpha} => Reject H0')
    else:
        print('Can NOT reject H0')


def test_condition_number():
    pass


def test_variance_inflation_factor():
    pass


def test_breusch_pagan(s, exog, alpha=0.01, **kwargs):
    print('H0: series is homoskedastic', end='\n\n')
    print('Results of BP Test:')
    names = ['Lagrange multiplier statistic', 'p-value',
             'f-value', 'f p-value']
    bptest = sm.stats.het_breuschpagan(s, exog)
    bp_output = _pd.Series(bptest, index=names)
    print(bp_output, end='\n\n')
    pvalue = bptest[1]
    if pvalue < alpha:
        print(f'p-value {pvalue:.4f} is less alpha {alpha} => Reject H0')
    else:
        print('Can NOT reject H0')


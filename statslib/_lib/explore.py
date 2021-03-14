import matplotlib as _mpl
import statsmodels.api as _sm
import matplotlib.pyplot as plt
from statslib.utils.common import to_namedtuple as _to_namedtuple


def decompose_seasonal_stl(y, figsize=(8*1.6, 8), **kwargs):
    from statsmodels.tsa.seasonal import STL
    stl = STL(y, **kwargs)
    res = stl.fit()
    with _mpl.rc_context():
        _mpl.rc("figure", figsize=figsize)
        res.plot()
    plt.show()
    return res


def decompose_seasonal_trend_residual(s, figsize=(9 * 1.6, 9), **kwargs):
    decompose_result = _sm.tsa.seasonal_decompose(s, **kwargs)
    with _mpl.rc_context():
        _mpl.rc("figure", figsize=figsize)
        trend = decompose_result.trend
        seasonal = decompose_result.seasonal
        residual = decompose_result.resid
        decompose_result.plot()
        return _to_namedtuple(dict(zip(['trend', 'seasonal', 'residual', 'result'],
                                      [trend, seasonal, residual, decompose_result])))

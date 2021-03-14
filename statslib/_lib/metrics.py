import numpy as _np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mean_percentage_error(y_true, y_pred):
    y_true, y_pred = _np.array(y_true), _np.array(y_pred)
    return _np.mean((y_true - y_pred) / y_true)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = _np.array(y_true), _np.array(y_pred)
    return _np.mean(_np.abs((y_true - y_pred) / y_true))

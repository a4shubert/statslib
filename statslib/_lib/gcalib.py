from enum import Enum
import inspect


class CalibType(Enum):
    sm = 'statsmodels'
    sk = 'sklearn'


class GeneralCalibrator:
    def __init__(self, cf, kwargs=None):
        self.cf = cf
        self.kwargs = kwargs
        if 'sklearn' in cf.__module__:
            self.calib_type = CalibType.sk
        elif 'statsmodels' in cf.__module__:
            self.calib_type = CalibType.sm
        else:
            raise NotImplementedError(f'calibrator from package {cf.__module__} not accommodated yet')
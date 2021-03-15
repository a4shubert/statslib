import os
import numpy as np
import pandas as pd

from unittest import TestCase
from statslib._pathmap import TEST_FOLDER
from statslib.utils.dframe import to_pd_todatetime


class TransformationsTest(TestCase):
    def setUp(self) -> None:
        path = os.path.join(TEST_FOLDER, '../statslib/datasets', 'data/y.csv')
        df = pd.read_csv(path)
        df = to_pd_todatetime(df, 'day', day_only=True)
        self.y = df.set_index('day').squeeze().rename('y')

    def test_pct_change(self):
        from statslib._lib.transforms import pct_change
        for i in range(1, 10):
            f = pct_change(i)
            pd.testing.assert_series_equal(self.y, f.inv(f(self.y)).rename('y'))

    def test_log_return(self):
        from statslib._lib.transforms import log_return
        for i in range(1, 10):
            f = log_return(i)
            pd.testing.assert_series_equal(self.y, f.inv(f(self.y)).rename('y'))

    def test_difference_operator(self):
        from statslib._lib.transforms import difference_operator
        spec_dict = {(2, None, 0): [[-2, 1], [1, 2]],
                     (1, 1, 3): [[-1, -1, 1], [1, 3, 4]],
                     (2, 1, 3): [[-2, 1, -1, 2, -1], [1, 2, 3, 4, 5]]}

        for spec, inv_spec in spec_dict.items():
            d, D, s = spec
            f = difference_operator(d, D, s, inv_specification=inv_spec)
            pd.testing.assert_series_equal(self.y, f.inv(f(self.y)).rename('y'))

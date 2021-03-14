__import__('pkg_resources').declare_namespace(__name__)



from statslib._lib.cross_validation import CrossValidation
from statslib._lib.design_matrix import DesignMatrix
from statslib._lib.gcalib import GeneralCalibrator
from statslib._lib.gmodel import GeneralModel

from statslib._lib import stat_plots
from statslib._lib import stat_tests
from statslib._lib import transforms
from statslib._lib import metrics
from statslib._lib.datasets import datasets
from statslib._lib import explore

from statslib import utils

from statslib._smdt.smdt import SmartData
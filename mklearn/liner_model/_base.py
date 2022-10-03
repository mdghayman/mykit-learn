from abc import ABCMeta, abstractmethod
import numbers
import warnings

import numpy as np
import scipy.sparse as sp # Maybe redundant with sparse imported below.
from scipy import (
    linalg,
    optimize,
    sparse,
)
from scipy.sparse.linalg import lsqr
from scipy.special import expit
from joblib import Parallel
from numbers import Integral

from ..base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    MultiOutputMixin,
)
from ..preprocessing._data import _is_constant_feature
from ..utils._param_validation import (
    StrOptions,
    Hidden,
)
from ..utils import (
    check_array,
    check_random_state,
)
from ..utils.extmath import (
    safe_sparse_dot,
    _incremental_mean_and_var,
)
from ..utils.sparsefuncs import (
    mean_variance_axis,
    inplace_column_scale,
)
from ..utils._array_api import get_namespace
from ..utils._seq_dataset import (
    ArrayDataset32,
    CSRDataset32,
    ArrayDataset64,
    CSRDataset64,
)
from ..utils.validation import (
    FLOAT_DTYPES,
    check_is_fitted,
    _check_sample_weight,
)
from ..utils.fixes import delayed
from ..utils import


SPARSE_INTERCEPT_DECAY = 0.01


def _deprecate_normalize(normalize, default, estimator_name):
    pass # Probably irrelevant as deprecated, but look for _normalize variable in other functions.

def make_dataset(X, y, sample_weight, random_state=None):
    pass

def _preprocess_data(X, y, fit_intercept, normalize=False, copy=True, sample_weight=None, check_input=True):
    pass

def _rescale_data(X, y, sample_weight):
    pass

class LinearModel(BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def fit(self, X, y):
        pass

    def _decision_function(self, X):
        pass

    def predict(self, X):
        pass

    def _set_intercept(self, X_offset, y_offset, X_scale):
        pass

    def _more_tags(self):
        pass

class LinearClassifierMixin(ClassifierMixin):

    def decision_function(self, X):
        pass

    def predict(self, X):
        pass

    def _predict_proba_lr(self, X):
        pass

class SparseCoefMixin:

    def densify(self):
        pass

    def sparsify(self):
        pass

class LinearRegression(MultiOutputMixin, RegressorMixin, LinearModel):

    def __init__(self, *, fit_intercept=True, normalize="deprecated", copy_X=True, n_jobs=None, positive=False):
        pass

    def fit(self, X, y, sample_weight=None):
        pass

def _check_precomputed_gram_matrix(X, precompute, X_offset, X_scale, rtol=None, atol=1e-5):
    pass

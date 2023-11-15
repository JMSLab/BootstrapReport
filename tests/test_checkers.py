""" Test functions in checkers.py """

import numpy as np
import pytest
from scipy.stats import norm
import warnings
import BootstrapReport.checkers as checkers

def test_check_parameters():
    with pytest.raises(Exception):
        checkers.check_parameters(1, 1, 0)
    with pytest.raises(Exception):
        checkers.check_parameters(0, 1, 1)
    with pytest.raises(Exception):
        checkers.check_parameters(1, 0, 1)
    with pytest.raises(Exception):
        checkers.check_parameters(0, 0, 0)


def test_check_integration_bounds():
    def normal_pdf(x):
        return norm.pdf(x, 0, 1)

    with warnings.catch_warnings():
        checkers.check_integration_bounds(3, 1, 1, np.inf, normal_pdf)

    with pytest.warns(UserWarning):
        checkers.check_integration_bounds(0, 1, 1, 5, normal_pdf)
    with pytest.warns(UserWarning):
        checkers.check_integration_bounds(0.2, 1, 1, 1, normal_pdf)
    with pytest.warns(UserWarning):
        checkers.check_integration_bounds(0, 1, 1, np.inf, normal_pdf)

def test_check_bias_accuracy():
    with warnings.catch_warnings():
        checkers.check_bias_accuracy(0.05, 0, 1)
        checkers.check_bias_accuracy(0.05, 0.1, 1)
        checkers.check_bias_accuracy(0.05, 0.1, 2)

    with pytest.warns(UserWarning):
        checkers.check_bias_accuracy(0.05, 0.5, 2)


def test_check_randomization():
    assert checkers.check_randomization([1, 2, 3, 4, 5]) is None
    assert checkers.check_randomization([[1, 2, 3], [1, 2, 4]]) is None
    with pytest.warns(UserWarning):
        checkers.check_randomization([1, 2, 3, 4, 5, 5])
        

def test_check_seed():
    assert checkers.check_seed(1, 2) is None
    assert checkers.check_seed(1, 1.5) is None
    assert checkers.check_seed("a", "b") is None
    with pytest.raises(SystemExit):
        assert checkers.check_seed(1, 1) == SystemExit
        assert checkers.check_seed("1", "1") == SystemExit

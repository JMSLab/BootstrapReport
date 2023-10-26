""" Test functions in checkers.py """

import numpy as np
import pytest
from scipy.stats import norm
import BootstrapReport.checkers


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

    with pytest.warns(None) as record:
        checkers.check_integration_bounds(3, 1, 1, np.inf, normal_pdf)
    assert len(record) == 0

    with pytest.warns(None) as record:
        checkers.check_integration_bounds(0, 1, 1, 5, normal_pdf)
        checkers.check_integration_bounds(0.2, 1, 1, 1, normal_pdf)
        checkers.check_integration_bounds(0, 1, 1, np.inf, normal_pdf)
    assert len(record) == 3
    assert str(record[0].message) == "Warning: Consider increasing C \n or using the default infinite bounds."


def test_check_bias_accuracy():
    with pytest.warns(None) as record:
        assert checkers.check_bias_accuracy(0.05, 0, 1) is None
        assert checkers.check_bias_accuracy(0.05, 0.1, 1) is None
        assert checkers.check_bias_accuracy(0.05, 0.1, 2) is None
    assert len(record) == 0

    with pytest.warns(None) as record:
        assert checkers.check_bias_accuracy(0.05, 0.1, 2, tau=0.001) is None
    assert len(record) == 1


def test_check_randomization():
    assert checkers.check_randomization([1, 2, 3, 4, 5]) is None
    assert checkers.check_randomization([[1, 2, 3], [1, 2, 4]]) is None
    message = "Warning: At least two sets of random draws are identical."
    with pytest.warns(match=message):
        checkers.check_randomization([1, 2, 3, 4, 5, 5])
    with pytest.warns(match=message):
        checkers.check_randomization([[1, 2, 3], [1, 2, 3]])
        

def test_check_seed():
    assert checkers.check_seed(1, 2) is None
    assert checkers.check_seed(1, 1.5) is None
    assert checkers.check_seed("a", "b") is None
    with pytest.raises(SystemExit):
        assert checkers.check_seed(1, 1) == SystemExit
        assert checkers.check_seed("1", "1") == SystemExit

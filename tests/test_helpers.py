""" Test functions in helpers.py """

import warnings
from math import log10, floor
import numpy as np
from scipy import integrate
from scipy.stats import norm, uniform
import BootstrapReport.helpers as helpers

def round_to_2(x, sig=2):
    """ for testing purposes: round to the first two sig figs. e.g. 0.5001 -> 0.50 """
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def test_get_nested_list():
    assert helpers.get_nested_list(3) == [[], [], []]
    assert not helpers.get_nested_list(0)
    assert helpers.get_nested_list(1) == [[]]


def test_get_integration_bounds():
    assert helpers.get_integration_bounds(0, 1, 1) == (-1, 1)
    assert helpers.get_integration_bounds(1, 0, 0) == (1, 1)
    assert helpers.get_integration_bounds(1, 1, 1) == (0, 2)
    assert helpers.get_integration_bounds(1, 1, 2) == (-1, 3)


def test_get_rot_bandwidth():
    assert helpers.get_rot_bandwidth(1, 1) == (4 / 3) ** (1 / 5)
    assert helpers.get_rot_bandwidth(1, 0) == 0
    assert helpers.get_rot_bandwidth(10, 0) == 0
    # assert helpers.get_rot_bandwidth(0, 0) == 0  # Fails, gives ZeroDivisionError


def test_get_tvd():
    # Failure
    assert helpers.get_tvd(uniform(0, 1).pdf, uniform(50, 1).pdf, -np.inf, np.inf) != 0.99  # scipy error
    assert helpers.get_tvd(uniform(0, 1).pdf, uniform(50, 1).pdf, -100, 100) != 0.99  # scipy error
    # Edge cases
    assert integrate.quad(uniform.pdf, -1, 1) == (1.0, 1.1102230246251565e-14)
    assert helpers.get_tvd(uniform.pdf, uniform.pdf, -np.inf, np.inf) == 0.0
    assert round_to_2(helpers.get_tvd(uniform(0).pdf, uniform(1).pdf)) == 1
    assert round_to_2(helpers.get_tvd(uniform.pdf, uniform(5).pdf)) == 1
    assert round_to_2(helpers.get_tvd(uniform(0, 1).pdf, uniform(50, 1).pdf, 0, 51)) == 1
    assert helpers.get_tvd(norm.pdf, norm.pdf, -np.inf, np.inf) == 0.0
    assert round_to_2(helpers.get_tvd(norm.pdf, lambda x: norm.pdf(x, 100, 1), -100, 200)) == 1

    # Uniform distributions
    theta1, theta2 = 1, 2
    if theta2 > theta1:
        assert round_to_2(helpers.get_tvd(uniform(0, theta1).pdf, uniform(0, theta2).pdf, 0, 3)) \
               == (theta2 - theta1) / theta2
    else:
        warnings.warn("Warning: theta2 must be greater than theta1.", stacklevel=2)

    # Normal vs uniform distributions
    for theta, mu, sigma in [(10, -5, 3), (10, 0, 3), (10, 5, 3), (10, 10, 3), (10, 15, 3)]:
        uniform_dist = lambda x: uniform.pdf(x, loc = 0, scale = theta)
        norm_pdf = lambda x: norm.pdf(x, loc = mu, scale = sigma)
        norm_cdf = lambda x: norm.cdf(x, loc = mu, scale = sigma)
        a = mu - sigma * np.sqrt(-2 * np.log( (sigma * np.sqrt(2 * np.pi)) / theta ))
        b = mu + sigma * np.sqrt(-2 * np.log( (sigma * np.sqrt(2 * np.pi)) / theta ))

        if sigma >= theta / np.sqrt(2 * np.pi):
            tvd = 1 - (norm_cdf(theta) - norm_cdf(0))
        elif b <= 0:
            tvd = 1 - (norm_cdf(theta) - norm_cdf(0))
        elif a <= 0 and b < theta:
            tvd = (theta - b)/theta - (norm_cdf(theta) - norm_cdf(b))
        elif b < theta:
            tvd = (theta - b)/theta - (norm_cdf(theta) - norm_cdf(b)) + a/theta - (norm_cdf(a) - norm_cdf(0))
        elif a < theta and b >= theta:
            tvd = a/theta - (norm_cdf(a) - norm_cdf(0))
        elif a >= theta:
            tvd = 1 - (norm_cdf(theta) - norm_cdf(0)) 
        # Assert expected value is equal to get_tvd() calculation
        assert round_to_2(helpers.get_tvd(norm_pdf, uniform_dist)) == round_to_2(tvd)
        assert round_to_2(helpers.get_tvd(norm_pdf, uniform_dist, max_subdivisions = 100)) == round_to_2(tvd)

    
    # Normal distributions
    mu_success_list = ((0, 1), (0, 5), (0, 3), (0, 3.3), (0, 0.01), (0, 10))
    for mu1, mu2 in mu_success_list:
        assert round_to_2(helpers.get_tvd(lambda x: norm.pdf(x, mu1, 1), lambda x: norm.pdf(x, mu2, 1), -25, 25)) \
            == round_to_2(norm.cdf(mu2 / 2) - norm.cdf(- mu2 / 2))
    mu_fail_list = ((0, -1), (1, 5))
    for mu1, mu2 in mu_fail_list:
        assert round_to_2(helpers.get_tvd(lambda x: norm.pdf(x, mu1, 1), lambda x: norm.pdf(x, mu2, 1), -25, 25)) \
            != round_to_2(norm.cdf(mu2 / 2) - norm.cdf(- mu2 / 2)) # fails when mu1 != 0

def test_get_grid():
    assert np.array_equal(helpers.get_grid(0, 1, -1, 1), np.array([0.0, 0.0])) is True
    assert np.array_equal(helpers.get_grid(1, -1, 1, 2), np.array([0.1, 1.0, 10.0])) is True
    assert np.array_equal(helpers.get_grid(1, -1, 1, 1), np.array([0.1, 10.0])) is True

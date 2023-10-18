""" This file contains checking functions """

import sys
import warnings
import numpy as np
from scipy import integrate
from scipy.stats import norm
import pandas as pd

def check_parameters(num_gridpoints, num_sets, bounds_of_integration):
    if num_gridpoints < 1:
        raise Exception("Requires number of gridpoints to be > 1")
    if num_sets < 1:
        raise Exception("Requires number of sets to be > 1")
    if bounds_of_integration == 0:
        raise Exception("Bounds of integration cannot be scaled by 0")


def check_integration_bounds(tvd, estimate, se, bounds_of_integration, kde_pdf_replicates):
    """ check if the integration bounds (C) is large enough """

    def integrand(x):
        if x == np.inf or x == -np.inf:
            return 0
        else:
            return kde_pdf_replicates(x)

    LHS = tvd
    rbound_a = estimate - (bounds_of_integration * se)
    rbound_b = estimate + (bounds_of_integration * se)
    RHS = integrate.quad(integrand, -np.inf, rbound_a)[0] + 1 - integrate.quad(integrand, -np.inf, rbound_b)[0]
    if LHS <= RHS:
        warnings.warn("Warning: Consider increasing C \n or using the default infinite bounds.", stacklevel=2)


def check_boundary(band_at_boundary):
    """ check if the selected bandwidth is at or close to the endpoints of the grid
    :param band_at_boundary: boolean value for if band is at boundary
    :return: warning if optimal bandwidth is close to or on the endpoints of the grid. otherwise, do nothing.
    """
    if band_at_boundary:
        warnings.warn("Warning: Chosen bandwidth is at or near the [lower/upper] limit of the grid after adaptive search. "
                      "Increase the max grid expansion for more accuracy.",
                      stacklevel=2)
    else:
        pass


def check_bias_accuracy(t_star_star, bias, bias_corrected_tvd_estimate, tau0 = 0.1, tau1 = 0.001):
    """ check if accuracy of bias_value correction appears low """
    accuracy_magnitude = abs(t_star_star - bias)
    accuracy = accuracy_magnitude / bias_corrected_tvd_estimate

    if accuracy >= tau0 and accuracy_magnitude >= tau1:
        warnings.warn("Warning: Accuracy of bias_value \n correction appears low. Consider increasing number of "
                      "sets of replicates used for calculation.", stacklevel=2)


def check_randomization(tvd_list):
    """ check if any sets of random draws are identical """
    list_of_tvds = []
    for tvd in tvd_list:
        if tvd in list_of_tvds:
            warnings.warn("Warning: At least two sets of random draws are identical." , stacklevel=2)
        list_of_tvds.append(tvd)


def check_seed(first_seed, second_seed):
    if first_seed == second_seed:
        return sys.exit("ERROR: the seed for the first set of draws cannot equal the seed for the second set of draws.")
    else:
        pass
    
def check_initial_values(x):
    x = np.array(x)
    if x.shape != (2,):
        raise ValueError("`init_values` must be array-like with shape (2,)")
    if not all(isinstance(i, (int, np.integer, float)) for i in x):
        raise TypeError("`init_values` must be integer or float")
    if not all(i==i for i in x):
        raise ValueError("`init_values` cannot contain NaN values")
    if (x[1] <= 0) or (x[1] >= np.inf):
        raise ValueError("`init_values` must specify a value for the standard deviation in (0, inf)")

def check_optimization_bounds(bounds):
    bounds = np.array(bounds)
    if bounds.shape != (2,2):
        raise ValueError("`bounds` must be array-like with shape (2,2)")
    if not all(isinstance(i, (int, np.integer, float)) for i in bounds.flatten()):
        raise TypeError("`bounds` must be integer or float")
    if not all(i==i for i in bounds.flatten()):
        raise ValueError("`bounds` cannot contain NaN values")
    if bounds[1, 0] <= 0:
        raise ValueError("`bounds` must specify a range for standard deviation inside (0, inf]")

def check_density(mean, std, bounds):
    norm_oi = norm(loc = mean, scale = std)
    cd = integrate.quad(norm_oi.pdf, bounds[0], bounds[1])
    if cd[0] <= 0.99:
        message =   'Accuracy of numerical integrals appears poor. Consider starting with a ' + \
                    'larger guess for the standard deviation.'
        warnings.warn(message, UserWarning)

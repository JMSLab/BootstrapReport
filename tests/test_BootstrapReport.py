""" Test functions in ObjectOfInterest.py """

import numpy as np, pandas as pd
from scipy.stats import norm, uniform
from scipy.optimize import minimize
import pytest
from BootstrapReport.BootstrapReport import ObjectOfInterest
import BootstrapReport.helpers as helpers
import test_helpers

def test_sk_distance():
    test_replicates = pd.read_csv('examples/gamma_replicates.csv')['replicate_value'].values
    estimate, standard_error = 0, 1
    test = ObjectOfInterest(estimate = estimate, se = standard_error, replicates = test_replicates)
    test.pp_plot()
    
    assert test.neg_dist <= 1 and test.neg_dist >= 0
    assert test.pos_dist <= 1 and test.pos_dist >= 0

def test_density_plot():
    test_replicates = pd.read_csv('examples/gamma_replicates.csv')['replicate_value'].values
    estimate, standard_error = 0, 1
    test = ObjectOfInterest(estimate = estimate, se = standard_error, replicates = test_replicates)
    
    # Test function breaks when there is no bandwidth
    with pytest.raises(ValueError):
        test.density_plot()
    
    test.get_bias_corrected_tvd(num_sets = 2)
    test.density_plot()

def test_crossings():
    test_replicates = pd.read_csv('examples/gamma_replicates.csv')['replicate_value'].values
    estimate, standard_error = 0, 1
    test = ObjectOfInterest(estimate = estimate, se = standard_error, replicates = test_replicates)
    # Test function makes it to `savefig` when plotting outfile
    with pytest.raises(AttributeError):
        test.get_crossings(outfile = True)
    test = ObjectOfInterest(estimate = estimate, se = standard_error, \
                            replicates = norm.ppf(np.linspace(0.01, 1, 100), loc = estimate, scale = standard_error))
    test.get_crossings()
    assert test.crossings == 0
    
    for estimate in [-10, 10]:
        test = ObjectOfInterest(estimate = estimate, se = standard_error, \
                                replicates = norm.ppf(np.linspace(0.01, 1, 100), loc = -estimate, scale = standard_error))
        test.get_crossings()
        assert test.crossings == 1
    

def test_pp_plot():
    test_replicates = pd.read_csv('examples/gamma_replicates.csv')['replicate_value'].values
    estimate, standard_error = 12.53787813379, 41.35729698836584
    test = ObjectOfInterest(estimate = estimate, se = standard_error, replicates = test_replicates)
    
    test.pp_plot(alpha = 0.01)

    # Test function breaks when incorrect argument types are given
    with pytest.raises(AttributeError):
        test.pp_plot(outfile = True)
    with pytest.raises(TypeError):
        test.pp_plot(alpha = 'test')

def test_sk_min_normal():
    tol = 1e-1
    par = np.array([[0,1], [0,10], [-10,1]])
    R = 1000
    
    np.random.seed(1)
    for p in par:
        df = np.random.normal(p[0], p[1], R).tolist()
        test = ObjectOfInterest(p[0], p[1], replicates=df)
                    
        res = test.get_sk_min()
        assert np.isclose(res.x[0], p[0], atol=tol, rtol=tol)
        assert np.isclose(res.x[1], p[1], atol=tol, rtol=tol)
        
        res1 = test.get_sk_min(init_values="ESTIMATES")
        assert np.isclose(res1.x[0], res.x[0], atol=tol, rtol=tol)
        assert np.isclose(res1.x[1], res.x[1], atol=tol, rtol=tol)
      
        res = test.get_sk_min(init_values="REPLICATES")
        assert np.isclose(res.x[0], p[0], atol=tol, rtol=tol)
        assert np.isclose(res.x[1], p[1], atol=tol, rtol=tol)
        
    assert hasattr(test, "skmin_mean")
    assert hasattr(test, "skmin_sd")
    assert hasattr(test, "skmin_solveroutput")
    assert hasattr(test, "skmin")
    
def test_sk_min_warnings():
        
    par = np.array([0,1])
    R = 1000
    df = np.random.normal(par[0], par[1], R).tolist()
    test = ObjectOfInterest(par[0], par[1], replicates=df)    
    test.get_sk_min()
    
    with pytest.raises(ValueError):
        test.get_sk_min(bounds=((.5, 1.5), (0, 2)))
    with pytest.warns(UserWarning):
        test.get_sk_min(bounds=((-1, 1), (0.1, 0.2)))

# TV tests:
def test_bandwidth_to_se_ratio():
    test_replicates = pd.read_csv('examples/gamma_replicates.csv')['replicate_value'].values
    estimate1, se1 = 0, 1
    test1 = ObjectOfInterest(estimate1, se1, replicates=test_replicates)
    best_bandwidth1, tv_at_best_bwidth1 = test1.get_bias_corrected_tvd(num_sets=2)[1:3]
    estimate2, se2 = 0, 10
    test2 = ObjectOfInterest(estimate2, se2, replicates=test_replicates)
    best_bandwidth2, tv_at_best_bwidth2 = test2.get_bias_corrected_tvd(num_sets=2)[1:3]

    assert test_helpers.round_to_2(tv_at_best_bwidth1) == test_helpers.round_to_2(tv_at_best_bwidth2)
    assert test_helpers.round_to_2(best_bandwidth1) == test_helpers.round_to_2(best_bandwidth2)

def test_get_bias():
    test_replicates = pd.read_csv('examples/gamma_replicates.csv')['replicate_value'].values
    estimate, standard_error = 2, 1
    test = ObjectOfInterest(estimate=estimate, se=standard_error, replicates=test_replicates)
    target = 0.14286
    assert np.isclose(test.get_bias(num_replicates=10, best_bandwidth=1, implied_normal_pdf=lambda x: norm.pdf(x, 2, 1),
                          lbound=-50, rbound=50, num_sets=5, second_seed=11), target, atol = 0.00001)

def test_tv_min_uniform():
    tol = 1e-3
    theta = np.array([0.01, 500])
    c = 8.83739357236942583899741126
    
    for t in theta:
        
        def objective(inputs):
            def pdf_from_unif(x):
                return uniform.pdf(x, loc=0, scale=t)
            def pdf_from_normal(x):
                return norm.pdf(x, inputs[0], inputs[1])
            return helpers.get_tvd(pdf_from_unif, pdf_from_normal)

        x0 = (t/2, t/3)
        bounds = ((0, t), (0, t)) 
        res = minimize(objective, x0, bounds = bounds)
        assert np.isclose(res.x[0], t/2, atol=tol, rtol=tol) 
        assert np.isclose(res.x[1], t/np.sqrt(c), atol=tol, rtol=tol)
        
def test_tv_min_warnings():
        
    rep = pd.read_csv('examples/gamma_replicates.csv')['replicate_value'].values
    test = ObjectOfInterest(3.329153904463066, 2.0256286444979983, replicates=rep)    
    test.best_bandwidth_value = 0.24656
    
    with pytest.raises(ValueError):
        test.get_tv_min(optimization_bounds=((10, 20), (0, 2)))
    with pytest.warns(UserWarning):
        test.get_tv_min(optimization_bounds=((-1, 1), (0.1, 0.2)))
    with pytest.warns(UserWarning):
        test.get_tv_min(init_values=(1e-20, 1e-20))
    

def test_check_initial_values():
    
    rep = pd.read_csv('examples/gamma_replicates.csv')['replicate_value'].values
    test = ObjectOfInterest(3.329153904463066, 2.0256286444979983, replicates=rep)    
    test.best_bandwidth_value = 0.24656
    
    def test_message(error, arg):
        with pytest.raises(error):
            test.get_tv_min(init_values=arg)

    test_message(TypeError, np.array(["a", 1]))
    test_message(ValueError, 1)
    test_message(ValueError, [1, 2, 3])
    test_message(ValueError, np.array([[1, 2], [1, 3]]))
    test_message(ValueError, (np.nan, 1))
    test_message(ValueError, (0, np.inf))
    test_message(ValueError, (0, 0))
    

def test_check_optimization_bounds():
    
    rep = pd.read_csv('examples/gamma_replicates.csv')['replicate_value'].values
    test = ObjectOfInterest(3.329153904463066, 2.0256286444979983, replicates=rep)    
    test.best_bandwidth_value = 0.24656
    
    def test_message(error, arg):
        with pytest.raises(error):
            test.get_tv_min(optimization_bounds=arg)
        
    test_message(TypeError, np.array([["a", 1], [0, 1]]))
    test_message(ValueError, 1)
    test_message(ValueError, ((-1, 1), (0, 20), (0, 2))) 
    test_message(ValueError, np.array([[0, np.nan], [0, 1]]))
    test_message(ValueError, np.array([[-1, 1], [-1, 1]]))

def test_tv_min_normal():
    tol = 1e-1
    rep = pd.read_csv('examples/gamma_replicates.csv')['replicate_value'].values
    test = ObjectOfInterest(3.329153904463066, 2.0256286444979983, replicates=rep)    
    test.best_bandwidth_value = 0.24656
    test.get_tv_min()

    res_est = test.get_tv_min(init_values="ESTIMATES")
    res_rep = test.get_tv_min(init_values="REPLICATES")
    assert np.isclose(res_est.x[0], res_rep.x[0], atol=tol, rtol=tol)
    assert np.isclose(res_est.x[1], res_rep.x[1], atol=tol, rtol=tol)

    assert hasattr(test, "tvmin_mean")
    assert hasattr(test, "tvmin_sd")
    assert hasattr(test, "tvmin_solveroutput")
    assert hasattr(test, "tvmin")

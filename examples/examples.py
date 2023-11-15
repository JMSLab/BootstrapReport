""" Test examples """
import sys, os, re
sys.path.append('/'.join(re.split('/|\\\\', os.path.dirname( __file__ ))[0:-1]) + '/src/BootstrapReport')
import timeit
from BootstrapReport import ObjectOfInterest
import pandas as pd

def gamma_example():
    indir = 'examples'
    df_rep = pd.read_csv(f'{indir}/gamma_replicates.csv')
    df_est = pd.read_csv(f'{indir}/gamma_estimate.csv')
    replicates = df_rep['replicate_value'].values
    estimate, std_err = df_est.at[0, 'estimate'], df_est.at[0, 'std_err']

    start = timeit.default_timer()
    ex_object = ObjectOfInterest(estimate = estimate, se = std_err, replicates = replicates)
    ex_object.get_bias_corrected_tvd(num_sets=2, detail=True)

    # Returns True if attribute exists
    hasattr(ex_object, "estimate")
    ex_object.pp_plot(outfile = None)
    ex_object.get_bias_corrected_tvd(num_sets = 2, detail = False)
    ex_object.density_plot(outfile = None)
    ex_object.get_crossings(outfile = None)
    ex_object.get_tv_min()
    
    ### Vertical-distance-minimizing normal approximation to bootstrap replicates
    ex_object.get_sk_min()
    stop = timeit.default_timer()

    print('Runtime for gamma example (in seconds): ', stop - start)

def normal_example():
    indir = 'examples'
    df_rep = pd.read_csv(f'{indir}/normal_replicates.csv')
    df_est = pd.read_csv(f'{indir}/normal_estimate.csv')
    replicates = df_rep['replicate_value'].values
    estimate, std_err = df_est.at[0, 'estimate'], df_est.at[0, 'std_err']

    start = timeit.default_timer()
    ex_object = ObjectOfInterest(estimate = estimate, se = std_err, replicates = replicates)
    ex_object.get_bias_corrected_tvd(num_sets=2, detail=True)

    # Returns True if attribute exists
    hasattr(ex_object, "estimate")
    ex_object.pp_plot(outfile = None)
    ex_object.get_bias_corrected_tvd(num_sets = 2, detail = False)
    ex_object.density_plot(outfile = None)
    ex_object.get_crossings(outfile = None)
    
    ### Vertical-distance-minimizing normal approximation to bootstrap replicates
    ex_object.get_sk_min()
    stop = timeit.default_timer()

    print('Runtime for normal example (in seconds): ', stop - start)

if __name__ == '__main__':
    normal_example()
    gamma_example()

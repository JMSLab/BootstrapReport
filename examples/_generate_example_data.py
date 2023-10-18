""" Create example datasets"""
import numpy as numpy
import pandas as pd
from scipy.stats import gamma, norm, uniform
import numpy as np

def main():
    outdir = 'source/lib/BootstrapTV/examples'
    gamma_rep, gamma_est = generate_example_dataset('gamma')
    df_gamma_rep, df_gamma_est = pd.DataFrame(columns = ['replicate_value']), pd.DataFrame(columns = ['estimate', 'std_err'])
    df_gamma_rep['replicate_value'], df_gamma_est['estimate'] = gamma_rep, [gamma_est]
    df_gamma_est['std_err'] = np.std(gamma_rep, ddof = 1)
    df_gamma_rep.index += 1
    df_gamma_rep.to_csv(f'{outdir}/gamma_replicates.csv', index_label = 'replicate_number')
    df_gamma_est.to_csv(f'{outdir}/gamma_estimate.csv', index = False)

    normal_rep, normal_est = generate_example_dataset('normal')
    df_normal_rep, df_normal_est = pd.DataFrame(columns = ['replicate_value']), pd.DataFrame(columns = ['estimate', 'std_err'])
    df_normal_rep['replicate_value'], df_normal_est['estimate'] = normal_rep, [normal_est]
    df_normal_est['std_err'] = [np.std(normal_rep, ddof = 1)]
    df_normal_rep.index += 1
    df_normal_rep.to_csv(f'{outdir}/normal_replicates.csv', index_label = 'replicate_number')
    df_normal_est.to_csv(f'{outdir}/normal_estimate.csv', index = False)


def generate_example_dataset(name):
    """ Generates example data used in BootstrapTV library
    :param name: Name of the dataset. Either 'normal' for normal mean
        bootstrap replicates, or 'gamma' for gamma mean bootstrap replicates
    :return: Replicates and estimate for the data
    """
    np.random.seed(10042002)
    rng = np.random.default_rng(seed = 10042002)
    if name == 'normal':
        num_rep, mean, std = 100, 0.74, 0.086
        raw_data = norm.rvs(size = num_rep, loc = mean, scale = np.sqrt(num_rep) * std)
        replicates = get_replicates(raw_data)
        estimate = np.median(raw_data)
    elif name == 'gamma':
        num_rep, variance, mean = 499, 0.186, 2.189
        raw_data = gamma.rvs((mean**2)/(variance * num_rep), scale = (variance * num_rep)/mean, size = 50)
        replicates = get_replicates(raw_data, num_rep = num_rep)
        estimate = np.average(raw_data, weights = uniform.rvs(size = 50))
    elif name == 'ratio':
        num_rep, numer_std, mean = 1000, 1, 0
        replicates = rng.normal(size = num_rep, loc = mean, scale = numer_std)/ \
            rng.normal(size = num_rep, loc = 1, scale = 0.35 * numer_std)
        estimate = np.quantile(replicates, 0.25, method = 'normal_unbiased')
    return replicates, estimate

def get_replicates(data, num_rep = None):
    """ Generates bootstrap replicates for the sample mean of data
    :param data: Raw data to be bootstrapped
    :param num_rep: Number of replicates to generate 
    :return: A numpy array containing the replicates
    """
    if num_rep is None:
        num_rep = data.shape[0]
    replicates = np.empty((0, 1))

    for rep in range(num_rep):
        num_draws = data.shape[0]
        draws = np.random.choice(data, num_draws, replace = True)
        replicate = np.mean(draws)
        replicates = np.append(replicates, replicate)
    
    return replicates

if __name__ == '__main__':
    main()

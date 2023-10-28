""" This file contains helper functions """

import numpy as np
from scipy import integrate, stats
import checkers
from matplotlib import pyplot as plt
import matplotlib as mpl

def get_nested_list(num_sublists) -> list:
    """ create a list of empty lists
    :param num_sublists: number of sublists to be created inside main list
    :return: list of lists
    """
    main_list = []
    for _ in range(num_sublists):
        sublist = []
        main_list.append(sublist)
    return main_list


def get_integration_bounds(estimate, bounds_of_integration, se):
    """ calculate left and right boundaries for integration """
    if bounds_of_integration != np.inf:
        lbound = estimate - (bounds_of_integration * se)
        rbound = estimate + (bounds_of_integration * se)
    else:
        lbound = -np.inf
        rbound = np.inf
    return lbound, rbound


def get_rot_bandwidth(num_replicates, se):
    """ calculate rot_se via rule of thumb """
    return (4 / (3 * num_replicates)) ** (1 / 5) * se


def get_kde(data, bandwidth):
    """ conduct kernel density estimation on a set of data points with the inputted bandwidth
    :param data: data to be smoothed
    :param bandwidth: bandwidth of kernel
    :return: estimated continuous probability density function
    """
    pdf_from_kde = stats.gaussian_kde(data, bw_method=bandwidth)

    def pdf_return(x):
        return pdf_from_kde(x)[0]

    return pdf_return


def get_tvd(pdf1, pdf2, lbound=-np.inf, rbound=np.inf, max_subdivisions = 50):
    """ calculate TVD between two continuous probability distribution functions
    :param pdf1: the first probability distribution functions
    :param pdf2: the second probability distribution functions
    :param lbound: left integration bound
    :param rbound: right integration bound
    :param max_subdivisions: number of subdivisions allowed in integrate function
    :return: total variation distance
    """

    def integrand(x):
        return abs(pdf1(x) - pdf2(x))

    distance = integrate.quad(integrand, lbound, rbound, limit=max_subdivisions)
    return 0.5 * distance[0]


def get_grid(rot_se, max_grid, min_grid, num_gridpoints):
    """ return the grid of bandwidths over which to search
    :param rot_se: calculated by scaling the standard error of the point estimate by the ROT
    :param min_grid: lower range of the gridpoints, expressed as a power-of-10 multiple of the ROT bandwidth
    :param max_grid: upper range of the gridpoints, expressed as a power-of-10 multiple of the ROT bandwidth
    :param num_gridpoints: number of gridpoints (minus one)
    :return: gridpoints of values over which to search for the optimal bandwidth
    """
    gridpoints = []
    for i in range(num_gridpoints + 1):
        gridpoints.append(10.0 ** (min_grid * (float(i) / num_gridpoints)
                                   + max_grid * ((float(num_gridpoints - i)) / num_gridpoints)) * rot_se)
    return gridpoints


def get_bandwidth_helper(estimate, se, num_sets, gridpoints, num_replicates, lbound, rbound, first_seed, max_subdivisions):
    """ calculate lists of TVD between the PDF N(point estimate, se) estimated from the draws and the
        precise PDF of N(point estimate, se)
    :param estimate: point estimate
    :param se: standard error of the point estimate
    :param num_sets: number of iterations
    :param gridpoints: gridpoints of values
    :param num_replicates: number of replicates
    :param lbound: left integration bound
    :param rbound: right integration bound
    :param first_seed: set seed for set of draws
    :param max_subdivisions: number of subdivisions allowed in integrate function
    :return: lists of TVD and exact PDF for the normal implied by the estimate and SE
    """

    def implied_normal_pdf(x):
        return stats.norm.pdf(x, estimate, se)

    tvd_lists = get_nested_list(len(gridpoints))
    np.random.seed(first_seed)
    for _ in range(num_sets):
        draws_from_implied_normal = np.random.normal(estimate, se, num_replicates)
        index = 0
        for bandwidth in gridpoints:
            pdf_from_kde = get_kde(draws_from_implied_normal, bandwidth)
            tvd = get_tvd(pdf_from_kde, implied_normal_pdf, lbound, rbound, max_subdivisions)
            tvd_lists[index].append(tvd)
            index += 1

    checkers.check_randomization(tvd_lists)

    return tvd_lists, implied_normal_pdf

def get_sk_dist(rep, normal, sep = False):
    ''' calculates the sk_distance between a sorted array of replicates and a normal distribution
    :param rep: sorted array of replicates
    :param normal: instance of scipy.stats.norm
    :param sep: sep = True then return positive and negative distance separately. Otherwise return
        sk distance
    '''
    pos_dist, neg_dist = 0, 0
    num_rep = rep.shape[0]
    step = 0
    for i in range(len(rep)):
        at_norm = normal.cdf(rep[i])
        left_lim = step - at_norm
        if left_lim < neg_dist:
            neg_dist = left_lim
        step = (i + 1)/num_rep
        right_lim = step - at_norm
        if right_lim > pos_dist:
            pos_dist = right_lim
    
    if sep:
        return -neg_dist, pos_dist
    else:
        return -neg_dist + pos_dist

def select_bandwidth(rot_se, max_grid, min_grid, num_gridpoints, 
                     estimate, se, num_sets, num_replicates, lbound, rbound, first_seed,
                     num_expansions, max_subdivisions):
    """ Finds the optimal bandwidth and then checks if bandwidth is on the edge of the grid. 
    If the best bandwidth is on the edge, it expands the grid and tries again.
    :param rot_se: calculated by scaling the standard error of the point estimate by the ROT
    :param max_grid: upper range of the gridpoints, expressed as a power-of-10 multiple of the ROT bandwidth
    :param min_grid: lower range of the gridpoints, expressed as a power-of-10 multiple of the ROT bandwidth
    :param num_gridpoints: number of gridpoints (minus one)
    :param estimate: point estimate
    :param se: standard error of the point estimate
    :param num_sets: number of iterations
    :param num_replicates: number of replicates
    :param lbound: left integration bound
    :param rbound: right integration bound
    :param first_seed: seed for first set of draws
    :param num_expansions: maximum number of grid expansions
    :param max_subdivisions: number of subdivisions allowed in integrate function
    :return implied_normal_pdf: exact PDF for the normal implied by the estimate and SE, 
    est bandwidth, index of the best bandwidth, list of TVDs, number of times grid was expanded in adaptive search.
    """
    
    band_at_boundary = True
    counter = 1

    while band_at_boundary == True and counter <= num_expansions:
        
        max_grid_scaled = max_grid * counter
        min_grid_scaled = min_grid * counter
        num_gridpoints_scaled = num_gridpoints * counter
        
        gridpoints = get_grid(rot_se, max_grid_scaled, min_grid_scaled, num_gridpoints_scaled)
        gridpoints.sort()
        boundary = [gridpoints[0], gridpoints[1], gridpoints[-2], gridpoints[-1]]
        
        tvd_lists, implied_normal_pdf = get_bandwidth_helper(estimate, se, num_sets, gridpoints,
                                                             num_replicates, lbound, rbound, first_seed, 
                                                             max_subdivisions)

        avg_tvd_list = []
        for b_list in tvd_lists:
            avg_tvd_list.append(np.mean(b_list))
        best_bandwidth_index = avg_tvd_list.index(min(avg_tvd_list))
        best_bandwidth = gridpoints[best_bandwidth_index]
                    
        if best_bandwidth in boundary:
            counter += 1
        else:
            band_at_boundary = False
            
    checkers.check_boundary(band_at_boundary)
    expansion_count = counter
    
    return implied_normal_pdf, best_bandwidth, best_bandwidth_index, avg_tvd_list, expansion_count

def plot_min_crossings(outfile, optimal_path, crossings, alpha, replicates, estimate, std, upper_cb, lower_cb, left_upper_cb, **kwargs):
    """ Create the plot for `get_crossings` of ObjectOfInterest
    :param outfile: destination for the plot
    :param optimal_path: nx2 numpy.array. First column contains x values, and second column contains y values of optimal path.
    :param crossings: number of crossings, or changes in direction
    :param alpha: 1 - alpha = confidence level for number of crossings 
    :param replicates: replicates of the object of interest
    :param estimate: estimate of the object of interest
    :param std: standard error of the object of interest
    :param upper_cb: function that returns the difference between the upper confidence bound and the cdf of the implied normal
    :param lower_cb: function that returns the difference between the lower confidence bound and the cdf of the implied normal
    :param left_upper_cb: function that returns the left-handed limit of the difference between the upper confidence bound 
        and the cdf of the implied normal
    """
    plt_set = {'fontsize': 18, 'legend_fontsize': 20, 'labelsize': 28, 'linecolor': '#f9665e', 'bandcolor': '#a8d9ed', 'linewidth': 2, 'dpi': 100}
    for key, value in kwargs.items():
        plt_set[key] = value
    mpl.rcParams['axes.spines.bottom'] = False
    plt.rcParams.update({'font.size': plt_set['fontsize']})
    plt.rcParams.update({'legend.fontsize': plt_set['legend_fontsize']})
    plt.rcParams.update({'axes.labelsize': plt_set['labelsize']})
    
    num_rep, x, y = len(replicates), optimal_path[-1, 0], optimal_path[-1, 1]
    plot_data = ('Num. crossings = %d' % crossings)
    props = dict(boxstyle = 'round, pad = 0.75, rounding_size = 0.3', facecolor = 'white', alpha = 0.7)
    plt.plot(optimal_path[:, 0], optimal_path[:, 1], color = plt_set['linecolor'], label = 'Optimal path', linewidth = plt_set['linewidth'])
    
    zero_null_rej = stats.norm.ppf(np.sqrt(np.log(2/alpha)/(2 * num_rep)), loc = estimate, scale = std)
    if zero_null_rej < replicates[0]:
        start = np.array([[zero_null_rej - 0.5 * std, 0]])
        xgrid = np.linspace(zero_null_rej, replicates[0], 25)[0:-1]
        start_path = [upper_cb(x) for x in xgrid]
        preline = np.column_stack((xgrid, start_path))
        preline = np.append(start, preline, axis = 0)
    else:
        start = np.array([[replicates[0] - 0.5 * std, 0], [replicates[0], 0]])
        preline = start

    zero_not_rej = stats.norm.ppf(1 - np.sqrt(np.log(2/alpha)/(2 * num_rep)), loc = estimate, scale = std)
    end = np.max([zero_not_rej, stats.norm.ppf(0.99, loc = estimate, scale = std), replicates[-1]])
    if y > 0:
        hits_top = stats.norm.ppf(1 - y, loc = estimate, scale = std)
        postline = np.array([[x, y], [hits_top, y]])
        xgrid = np.linspace(hits_top, end + std, 25)
        end_path = [left_upper_cb(x) for x in xgrid]
        postline = np.append(postline, np.column_stack((xgrid, end_path)), axis = 0)
    else:
        postline = np.array([[replicates[-1], optimal_path[-1, 1]], [end + 0.5 * std, 0], [end + std, 0]])
    
    plt.plot(preline[:, 0], preline[:, 1], color = plt_set['linecolor'], linewidth = plt_set['linewidth'])
    plt.plot(postline[:, 0], postline[:, 1], color = plt_set['linecolor'], linewidth = plt_set['linewidth'])
    x_axis = np.concatenate([np.linspace(replicates[r], replicates[r + 1], 10) for r in range(num_rep - 1)], axis = None)
    x_axis = np.append(np.linspace(preline[0, 0], replicates[0], 15), x_axis)
    x_axis = np.append(x_axis, np.linspace(replicates[-1], postline[-1, 0], 15))
    upper_band, lower_band = [upper_cb(x) for x in x_axis], [lower_cb(x) for x in x_axis]
    lbound, rbound = x_axis[0] + 0.25 * std, x_axis[-1] - 0.25 * std
    topbound = np.max(upper_band) + 0.025 * (np.max(upper_band) - np.min(lower_band))
    botbound = np.min(lower_band) - 0.025 * (np.max(upper_band) - np.min(lower_band))
    plt.xlim(lbound, rbound)
    plt.ylim(botbound, topbound)
    plt.hlines(y = botbound, xmin = lbound, xmax  = replicates[0], color = 'k', lw = 2, linestyle = (1.5, (1.5, 1)))
    plt.hlines(y = botbound, xmin = replicates[0], xmax  = replicates[-1], color = 'k', lw = 2)
    plt.hlines(y = botbound, xmin = replicates[-1], xmax = rbound, color = 'k', lw = 2, linestyle = (1.5, (1.5, 1)))
    plt.fill_between(x_axis, lower_band, upper_band, color = plt_set['bandcolor'], label = 'Confidence band', alpha = 0.25)
    plt.legend(edgecolor = 'k', loc = 'upper left')
    plt.text(rbound - 0.04 * (rbound - lbound), botbound + 0.07 * (topbound - botbound), plot_data, 
                fontsize = plt_set['legend_fontsize'], verticalalignment = 'bottom', horizontalalignment='right', bbox = props)
    plt.savefig(outfile, transparent = True, dpi = plt_set['dpi'])
    plt.clf()
    mpl.rcParams.update(mpl.rcParamsDefault)

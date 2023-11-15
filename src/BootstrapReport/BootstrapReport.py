""" Main file of the package """
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy.stats import norm
from scipy.optimize import minimize
import warnings
import checkers
import helpers
from _diagnostics import DiagnosticsMixin


class ObjectOfInterest(DiagnosticsMixin):

    def __init__(self, estimate, se, replicates):
        """
        :param estimate: point estimate
        :param se: standard error of the point estimate
        :param replicates: replicates from bootstrap
        """
        self.estimate = estimate
        self.se = se
        self.replicates = np.sort(replicates)
        self.bias_corrected_tv_value = None
        self.best_bandwidth_value = np.NaN
        self.bias_value = None
        self.crossings = None
        self.sk_dist = helpers.get_sk_dist(self.replicates, norm(loc = estimate, scale = se), sep = False)

    def get_bias_corrected_tvd(self, num_gridpoints=10, num_sets=100, min_grid=-1, max_grid=1,
                               bounds_of_integration=np.inf, first_seed=42, second_seed=11, 
                               num_expansions = 3, max_subdivisions = 50, detail=False):
        """
        :param num_gridpoints: number of gridpoints (minus one)
        :param num_sets: number of iterations
        :param min_grid: lower range of the gridpoints, expressed as a power-of-10 multiple of the ROT bandwidth
        :param max_grid: upper range of the gridpoints, expressed as a power-of-10 multiple of the ROT bandwidth
        :param bounds_of_integration: constant that scales the bounds of the integral in the TV formula
        :param first_seed: set seed for first set of draws
        :param second_seed: set seed for second set of draws
        :param num_expansions: maximum number of grid expansions
        :param detail: command to print additional information upon user request
        :return: bias_value-corrected TVD, selected bandwidth, avg TV with optimal bandwidth from first draw, estimated bias_value
        """

        checkers.check_parameters(num_gridpoints, num_sets, bounds_of_integration)
        checkers.check_seed(first_seed, second_seed)

        lbound, rbound = helpers.get_integration_bounds(self.estimate, bounds_of_integration, self.se)
        num_replicates = len(self.replicates)
        rot_se = helpers.get_rot_bandwidth(num_replicates, self.se)

        implied_normal_pdf, best_bandwidth, best_bandwidth_index, avg_tvd_list, expansion_count = helpers.select_bandwidth(rot_se, max_grid, min_grid, num_gridpoints, 
                                                                                                         self.estimate, self.se, num_sets, 
                                                                                                         num_replicates, lbound, rbound, first_seed, 
                                                                                                         num_expansions, max_subdivisions)
        
        num_gridpoints = num_gridpoints*expansion_count
        pdf_from_kde = helpers.get_kde(self.replicates, best_bandwidth)
        bias = self.get_bias(num_replicates, best_bandwidth, implied_normal_pdf, lbound, rbound, num_sets, second_seed)
        tvd = helpers.get_tvd(pdf_from_kde, implied_normal_pdf, lbound, rbound, max_subdivisions)
        bias_corrected_tv = tvd - bias

        checkers.check_integration_bounds(tvd, self.estimate, self.se, bounds_of_integration, pdf_from_kde)
        checkers.check_bias_accuracy(avg_tvd_list[best_bandwidth_index], bias, bias_corrected_tv, tau0=0.1)

        print("the bias_value-corrected total variation estimate = " + str(bias_corrected_tv) + ".")
        if detail:
            print("number of sets = " + str(num_sets) + ". grid fineness = " + str(
                num_gridpoints) + ". number of grid expansions = " + str(expansion_count) + 
                ". the selected bandwidth = " + str(best_bandwidth) + ".")
            print("t_b^* = " + str(avg_tvd_list[best_bandwidth_index]) + ". the bias_value = " + str(
                bias) + ".")
            
        self.expansion_count = expansion_count
        self.bias_corrected_tv_value = bias_corrected_tv
        self.best_bandwidth_value = best_bandwidth
        self.bias_value = bias
        return bias_corrected_tv, best_bandwidth, avg_tvd_list[best_bandwidth_index], bias

    def get_bias(self, num_replicates, best_bandwidth, implied_normal_pdf, lbound, rbound, num_sets, second_seed):
        """ calculate bias_value from a fresh set of draws
        :param num_replicates: number of replicates
        :param best_bandwidth: optimal bandwidth
        :param implied_normal_pdf: precise PDF for the normal implied by the estimate and SE
        :param lbound: left integration bound
        :param rbound: right integration bound
        :param num_sets: number of iterations
        :param second_seed: set seed for set of draws
        :return: bias_value
        """
        tvd_list = []
        np.random.seed(second_seed)
        for _ in range(num_sets):
            draws_from_implied_normal = np.random.normal(self.estimate, self.se, num_replicates)
            pdf_from_kde = helpers.get_kde(draws_from_implied_normal, best_bandwidth)
            tvd = helpers.get_tvd(pdf_from_kde, implied_normal_pdf, lbound, rbound)
            tvd_list.append(tvd)
        bias = np.mean(tvd_list)

        checkers.check_randomization(tvd_list)

        return bias
    
    def get_crossings(self, alpha = 0.05, outfile = None, **kwargs):
        """ calculate minimum changes in direction consistent with difference in CDFs. 
        values are normalized by the standard error
        :param alpha: 1 - alpha = confidence level for confidence bands
        :param outfile: path to output figure displaying algorithm
        """
        rep = self.replicates/self.se
        est = self.estimate/self.se
        num_rep, crossings = len(rep), 0
        rep_ecdf = lambda x: (1/num_rep) * np.sum(rep <= x)
        if alpha > 2 * np.exp(-2 * num_rep * rep_ecdf(rep[0])**2):
            raise ValueError('The target value of alpha is too large for the number of replicates.\n' +
                             'Please choose a smaller alpha')
        
        dkw_bot = lambda x: np.maximum(rep_ecdf(x) - np.sqrt(np.log(2/alpha)/(2 * num_rep)), 0)
        dkw_top = lambda x: np.minimum(rep_ecdf(x) + np.sqrt(np.log(2/alpha)/(2 * num_rep)), 1)
        lower_cb = lambda x: dkw_bot(x) - norm.cdf(x, loc = est, scale = 1)
        upper_cb = lambda x: dkw_top(x) - norm.cdf(x, loc = est, scale = 1)
        index_upper_over_1 = np.ceil(num_rep * (1 - np.sqrt(np.log(2/alpha)/(2 * num_rep))))
        def left_upper_cb(x):
            if x >= rep[int(index_upper_over_1 - 1)]:
                return upper_cb(x)
            else:
                return upper_cb(x) - 1/num_rep
    
        optimal_path = np.empty((0, 2))
        marker, x, y = None, -np.inf, 0

        x = rep[0]
        if lower_cb(x) <= 0 and upper_cb(x) >= 0:
            y = y
        elif upper_cb(x) < 0:
            y = left_upper_cb(x)
            marker = '-'
        optimal_path = np.append(optimal_path, [[x, y]], axis = 0)

        for t in range(1, num_rep):
            x = rep[t]
            if y >= left_upper_cb(x):
                if not outfile == None:
                    hits_top = norm.ppf(dkw_top(rep[t - 1]) - y, loc = est, scale = 1)
                    if hits_top < x:
                        optimal_path = np.append(optimal_path, [[hits_top, y]], axis = 0)
                        xgrid = np.linspace(hits_top, x, 20)[1:-1]
                        follows_ub_path = [upper_cb(x) for x in xgrid]
                        midline = np.column_stack((xgrid, follows_ub_path))
                        optimal_path = np.append(optimal_path, midline, axis = 0)
                y = left_upper_cb(x)
                if marker == '+':
                    crossings += 1
                marker = '-'
            elif y <= lower_cb(x):
                y = lower_cb(x)
                if marker == '-':
                    crossings += 1
                marker = '+'
            optimal_path = np.append(optimal_path, [[x, y]], axis = 0)
        
        if (marker == '-' and y < 0) or (marker == '+' and y > 0):
            crossings += 1
        self.crossings = crossings

        if not outfile == None:
            helpers.plot_min_crossings(outfile, optimal_path, self.crossings, alpha, rep, est,
                                       1, upper_cb, lower_cb, left_upper_cb, **kwargs)

    def pp_plot(self, confidence_band = True, alpha = 0.05, outfile = None, **kwargs):
        """ create the pp plot
        :param confidence_band: Boolean value of whether to include the confidence band in the plot
        :param outfile: location and name of file to be saved
        :param alpha: the upper bound for the probability that an ecdf plot of the normal
            approximation falls outside the shaded region
        """
        plt_set = {'fontsize': 18, 'legend_fontsize': 20, 'labelsize': 28, 'pointsize': 7, 'pointcolor': '#f9665e', 'bandcolor': '#a8d9ed', 'dpi': 100}
        for key, value in kwargs.items():
            plt_set[key] = value
        num_replicates = len(self.replicates)

        replicates_eval_normcdf = norm.cdf(self.replicates, self.estimate, self.se)
        replicate_ecdf = np.linspace(1/num_replicates, 1, num_replicates)

        dkw_xgrid = np.linspace(-0.05, 1.05, 200)
        dkw_lbound = [x - np.sqrt(np.log(2/alpha)/(2 * num_replicates)) for x in dkw_xgrid]
        dkw_ubound = [x + np.sqrt(np.log(2/alpha)/(2 * num_replicates)) for x in dkw_xgrid]

        self.neg_dist, self.pos_dist = helpers.get_sk_dist(self.replicates, norm(loc = self.estimate, scale = self.se), sep = True)

        plot_data = '\n'.join(
            ('Num. replicates = %d' % num_replicates,
             'Pos. distance = %.3f' % self.pos_dist,
             'Neg. distance = %.3f' % self.neg_dist))
        props = dict(boxstyle = 'round, pad = 0.75, rounding_size = 0.3', facecolor = 'white', alpha = 0.86)
        
        plt.rcParams.update({'font.size': plt_set['fontsize']})
        plt.rcParams.update({'legend.fontsize': plt_set['legend_fontsize']})
        plt.rcParams.update({'axes.labelsize': plt_set['labelsize']})
        
        plt.figure(figsize=(10, 10))
        if confidence_band == True:
            plt.fill_between(dkw_xgrid, dkw_lbound, dkw_ubound, color = plt_set['bandcolor'], label = 'Confidence band', alpha = 0.35)
        plt.scatter(replicates_eval_normcdf, replicate_ecdf, s = plt_set['pointsize'],
                    c = plt_set['pointcolor'], label='Bootstrap replicates')
        plt.xlabel("CDF of normal distribution")
        plt.ylabel("CDF of bootstrap distribution")
        plt.legend(edgecolor = 'k', loc = 'upper left')
        plt.axline((0, 0), (1, 1), color="black", linestyle=(0, (5, 5)))
        plt.text(0.52, 0.06, plot_data, fontsize = plt_set['legend_fontsize'], \
            verticalalignment = 'bottom', horizontalalignment='left', bbox = props)
        plt.ylim(0, 1)
        plt.xlim(0, 1)

        if not outfile == None:
            plt.savefig(outfile, transparent = True, dpi = plt_set['dpi'])
        plt.clf()
        mpl.rcParams.update(mpl.rcParamsDefault)
            
            
    def density_plot(self, bounds = None, bandwidth = None, outfile = None, **kwargs):
        """ creates a smoothed density plot of replicates and shows the plot or outputs the result to outfile
        :param bounds: a tuple or list of the bounds of the density plot | Optional
        :param bandwidth: the bandwidth that the replicates are evaluated at when taking the kernel density estimate | Optional
        :param outfile: location of the file to be saved | Optional
        """
        plt_set = {'fontsize': 18, 'legend_fontsize': 20, 'labelsize': 28, 'linecolor': '#f9665e', 'linewidth': 1, 'dpi': 100}
        for key, value in kwargs.items():
            plt_set[key] = value
        if not bandwidth:
            bandwidth = self.best_bandwidth_value
        if bounds != None:
            lbound, ubound = min(bounds), max(bounds)
        else:
            lbound, ubound = self.replicates[0] - 2*self.best_bandwidth_value, self.replicates[-1] + 2*self.best_bandwidth_value

        pdf_from_kde = helpers.get_kde(self.replicates, bandwidth)

        xgrid = np.linspace(lbound, ubound, len(self.replicates) * 100)
        density = [pdf_from_kde(x) for x in xgrid]

        plt.rcParams.update({'font.size': plt_set['fontsize']})
        plt.rcParams.update({'legend.fontsize': plt_set['legend_fontsize']})
        plt.rcParams.update({'axes.labelsize': plt_set['labelsize']})

        plt.xlim(lbound, ubound)
        plt.xlabel('Value of object of interest')
        plt.ylabel('Density')
        plt.plot(xgrid, density, linewidth = plt_set['linewidth'], color = plt_set['linecolor'])
        plt.plot([self.replicates[0], self.replicates[-1]], [0.0001, 0.0001], '|k', markeredgewidth = 1, label = 'Range of bootstrap replicates')
        plt.legend(loc = 'best', fontsize = 'x-small', markerscale = 0.75)
        
        if not outfile == None:
            plt.savefig(outfile, transparent = True, dpi = plt_set['dpi'])
        plt.clf()
        mpl.rcParams.update(mpl.rcParamsDefault)

    def get_tv_min(self, init_values = None, optimization_bounds = None, bounds_of_integration = np.inf):
        """
        :param init_values: Initial guess to be used with `scipy.optimize`. 
            Should be an array of real elements of size `(n,)`, with the first 
            element corresponding to the mean and the second element to the 
            standard deviation.
        :param optimization_bounds: Bounds to be used with `scipy.optimize`. Should be a 
            sequence of `(min, max)` pairs, with the first pair in the sequence
            corresponding to the mean and the second pair corresponding to the 
            standard deviation.
        :param bounds_of_integration: Multiple of `self.se` to use in calculating
            integration bounds for `integrate.quad` in TVD calculation. Should be a scalar.
        :return: Solver output.
        """
        if self.best_bandwidth_value is None:
            raise ValueError("self.best_bandwidth_value is None")
        
        pdf_from_kde = helpers.get_kde(self.replicates, self.best_bandwidth_value)
        lbound, rbound = helpers.get_integration_bounds(self.estimate, bounds_of_integration, self.se)
        def objective(inputs):
            def pdf_from_normal(x):
                return norm.pdf(x, inputs[0], inputs[1])
            return helpers.get_tvd(pdf_from_kde, pdf_from_normal, lbound, rbound)
        
        if init_values is None:
            init_values = "ESTIMATES"

        if isinstance(init_values, str):
            if init_values == "ESTIMATES":
                init_values = np.array([self.estimate, self.se])
            elif init_values == "REPLICATES":
                init_values = np.array([np.mean(self.replicates), np.std(self.replicates)])
        else:
            init_values = np.array(init_values)
        
        if optimization_bounds is None:
            R = len(self.replicates)
            lower = min(self.replicates)
            upper = max(self.replicates)
            optimization_bounds = np.array(((lower, upper), (1e-6, (upper - lower + 2 * \
                self.best_bandwidth_value * norm.ppf(1 - 1 / (4 * R))) / (2 * norm.ppf(0.5 + 1 / (4 * R))))))
        else:
            optimization_bounds = np.array(optimization_bounds)
        
        checkers.check_initial_values(init_values)
        checkers.check_optimization_bounds(optimization_bounds)
        checkers.check_density(init_values[0], init_values[1], (lbound, rbound))
        print("Initial values are: " + str(init_values))
        print("Optimization bounds are: " + str(optimization_bounds))

        res = minimize(objective, x0 = init_values, bounds = optimization_bounds)
        self.tvmin_mean = res.x[0]
        self.tvmin_sd = res.x[1]
        self.tvmin = res.fun
        self.tvmin_solveroutput = res

        if not res.success:
            warnings.warn("Warning: scipy.optimize did not exit successfully.")
            
        if any(np.isclose(res.x[0], optimization_bounds[0])):
            warnings.warn("Warning: minimizing value for mean is on the search boundary.")
            
        if any(np.isclose(res.x[1], optimization_bounds[1])):
            warnings.warn("Warning: minimizing value for standard deviation is on the search boundary. ")
        
        return res
        
        
    def get_sk_min(self, init_values = None, bounds = None):
        """
        :param init_values: Initial guess to be used with `scipy.optimize`. 
            Should be an array of real elements of size `(n,)`, with the first 
            element corresponding to the mean and the second element to the 
            standard deviation.
        :param bounds: Bounds to be used with `scipy.optimize`. Should be a 
            sequence of `(min, max)` pairs, with the first pair in the sequence
            corresponding to the mean and the second pair corresponding to the 
            standard deviation.
        :return: Solver output.
        """
            
        if init_values is None:
            init_values = "ESTIMATES"

        if isinstance(init_values, str):
            if init_values == "ESTIMATES":
                init_values = np.array([self.estimate, self.se])
            elif init_values == "REPLICATES":
                init_values = np.array([np.mean(self.replicates), np.std(self.replicates)])
            
        if bounds is None:
            lb = min(self.replicates)
            ub = max(self.replicates)
            bounds = np.array(((lb, ub), (1e-15, (ub - lb) / np.sqrt(2 * np.pi))))
        else:
            bounds = np.array(bounds)
            
        def get_sk_distance(inputs):
            return helpers.get_sk_dist(self.replicates, norm(loc = inputs[0], scale = inputs[1]), sep = False)
        
        checkers.check_initial_values(init_values)
        checkers.check_optimization_bounds(bounds)
        print("Initial values are: " + str(init_values))
        print("Optimization bounds are: " + str(bounds))
        
        res = minimize(get_sk_distance, x0 = init_values, method='nelder-mead', bounds = bounds)
        self.skmin_mean = res.x[0]
        self.skmin_sd = res.x[1]
        self.skmin = res.fun
        self.skmin_solveroutput = res
        
        if not res.success:
            warnings.warn("Warning: scipy.optimize did not exit successfully.")
            
        if any(np.isclose(res.x[0], bounds[0])):
            warnings.warn("Warning: minimizing value for mean is on the search boundary.")
            
        if any(np.isclose(res.x[1], bounds[1])):
            warnings.warn("Warning: minimizing value for standard deviation is on the search boundary. ")
        
        return res

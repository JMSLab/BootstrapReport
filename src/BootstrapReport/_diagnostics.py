import os
import seaborn as sns, numpy as np, pandas as pd
import helpers
import matplotlib.pyplot as plt
from scipy.stats import norm


class DiagnosticsMixin:
    
    def _heat_map(self, density = 50, outfile = False):
        """ create a heat map showing total variation distance as mean and standard deviation are varied
        :param mean_bounds: tuple or list of bounds for the mean
        :param sd_bounds: tuple or list of bounds for the standard deviation
        :param density: the density of the points in the heat map
        :outfile: location to be saved
        """
        sns.set_theme(font_scale=0.6)
        replicates = np.sort(self.replicates)
        mean_bounds = (self.estimate - 3*self.se, self.estimate + 3*self.se)
        sd_bounds = (self.se/5, 4*self.se)
    
        pdf_from_kde = helpers.get_kde(replicates, self.best_bandwidth_value)
        sigma_range, mu_range = np.linspace(min(sd_bounds), max(sd_bounds), density * 2), np.linspace(min(mean_bounds), max(mean_bounds), density)
        sigma_label, mu_label = ['%.3f' % sigma for sigma in sigma_range], ['%.3f' % mu for mu in mu_range]
    
        tvd_table = [[helpers.get_tvd(lambda x: norm.pdf(x, loc = mu, scale = sigma), pdf_from_kde) for sigma in sigma_range] for mu in mu_range]
        df = pd.DataFrame(tvd_table, columns = sigma_label, index = mu_label)
        ax = sns.heatmap(df, vmin = 0, vmax = 1)
        ax.invert_yaxis()
        ax.set_xlabel('σ')
        ax.set_ylabel('µ')
        plt.title('TVD with varied µ and σ')
    
        if outfile:
            plt.savefig(outfile, dpi=600)
            plt.clf()
        else:
            plt.figure().show()
            plt.close()
    
    def _comp_plot(self, outfile = False):
        """ creates a comparison plot for the density of the replicates and the optimized normal pdf
        :param outfile: location of the file to be saved
        """
        replicates = np.sort(self.replicates)
        bandwidth = self.best_bandwidth_value
        lbound, ubound = replicates[0] - 2*self.best_bandwidth_value, replicates[-1] + 2*self.best_bandwidth_value
    
        pdf_from_kde = helpers.get_kde(replicates, bandwidth)
    
        xgrid = np.linspace(lbound, ubound, len(self.replicates) * 25)
        density = [pdf_from_kde(x) for x in xgrid]
        normal = [norm.pdf(x, loc = self.tvmin_mean, scale = self.tvmin_sd) for x in xgrid]
    
        plt.xlim(lbound, ubound)
        plt.xlabel('Value of object of interest')
        plt.ylabel('Density')
        plt.plot(xgrid, density, linewidth = 1, color = 'r', label = 'kernel density plot')
        plt.plot(xgrid, normal, linewidth = 1, color = 'b', label = 'optimized normal')
        plt.plot([replicates[0], replicates[-1]], [0.0001, 0.0001], '|k', markeredgewidth = 1)
        plt.legend(loc = 'best', fontsize = 'x-small', markerscale = 0.75)
        
        if outfile:
            plt.savefig(outfile, dpi = 600)
            plt.clf()
        else:
            plt.figure().show()
            plt.close()
                
    def _profile_plot(self, density = 50, outfile = False):
        """ create 2-dimensional plots of total variation distance holding one of mean/standard deviation fixed and varying the other
        :param mean_bounds: tuple or list of bounds for the mean
        :param sd_bounds: tuple or list of bounds for the standard deviation
        :param density: the density of the points in the heat map
        :outfile: location to be saved
        """
        
        replicates = np.sort(self.replicates)
        mean_bounds = (self.tvmin_mean - 3*self.se, self.tvmin_mean + 3*self.se)
        sd_bounds = (self.se/5, 4*self.se)

        pdf_from_kde = helpers.get_kde(replicates, self.best_bandwidth_value)
        sigma_range, mu_range = np.linspace(min(sd_bounds), max(sd_bounds), density * 2), np.linspace(min(mean_bounds), max(mean_bounds), density)
        sigma_label, mu_label = ['%.3f' % sigma for sigma in sigma_range], ['%.3f' % mu for mu in mu_range]
        
        tvd_table = [helpers.get_tvd(lambda x: norm.pdf(x, loc = self.tvmin_mean, scale = sigma), pdf_from_kde) for sigma in sigma_range]
        df = pd.DataFrame(tvd_table, index = sigma_label)
        fig = plt.figure(figsize=(12, 10), dpi=600)

        ax = fig.add_subplot(111)
        ax.xaxis.set_major_locator(plt.MaxNLocator(50))
        ax.xaxis.set_tick_params(rotation=90)
        ax.set_xlabel('σ')
        ax.set_ylabel('TVD')
        ax.set_title('TVD with varied σ and µ = %.3f' % self.tvmin_mean)
        ax.plot(df.index.to_list(), df[0].to_list())
        
        if outfile: 
            plt.savefig('%s_sigma%s' % (os.path.splitext(outfile)), dpi = 600)
            plt.clf() 
        else: 
            plt.figure().show() 
            plt.close() 
            
        tvd_table = [helpers.get_tvd(lambda x: norm.pdf(x, loc = mu, scale = self.tvmin_sd), pdf_from_kde) for mu in mu_range]
        df = pd.DataFrame(tvd_table, index = mu_label)
        fig = plt.figure(figsize=(12, 10), dpi=600)

        ax = fig.add_subplot(111)
        ax.xaxis.set_major_locator(plt.MaxNLocator(50))
        ax.xaxis.set_tick_params(rotation=90)
        ax.set_xlabel('µ')
        ax.set_ylabel('TVD')
        ax.set_title('TVD with σ = %.3f and varied µ' % self.tvmin_sd)
        ax.plot(df.index.to_list(), df[0].to_list())
    
        if outfile: 
            plt.savefig('%s_mu%s' % (os.path.splitext(outfile)), dpi = 600)
            plt.clf() 
        else: 
            plt.figure().show() 
            plt.close() 
            

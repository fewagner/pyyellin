import csv
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


class Limit:
    """
    A class to calculate limits for dark matter cross-sections with Yellin's Optimum Interval Method.
    """

    def __init__(self):

        self.cl = 0.
        self.data = []
        self.corresponding_cdf = []
        self.table = []
        self.sigmas = []
        self.mus = []
        self.cdf = []
        self.efficiencies = []
        self.eff_grids = []
        self.acc_regions = []
        self.signal_models = []
        self.resolutions = []
        self.thresholds = []
        self.exposures = []

    # ------------------------------------------------
    # User API
    # ------------------------------------------------

    def add_sigmas_and_mus(self, sigmas: list, mus: list):
        """
        Add sigmas and corresponding mus.

        :param sigmas: Cross-sections.
        :param mus: Expected number of events.
        :return: None
        """
        self.sigmas = sigmas
        self.mus = mus
        return

    def add_cdf(self, cdf):
        """
        Add cumulative density function.

        :param cdf: Cumulative density function.
        :return: None
        """
        self.cdf = cdf
        return

    def set_confidence_level(self, cl: float):
        """
        Set confidence level.

        :param cl: Confidence level.
        :return: None
        """
        self.cl = cl
        return

    def get_limit(self):
        """
        Calculate the limit for the cross-section of dark matter particles using Yellin's Optimum Interval Method.

        :return: None
        """

        k_largest_intervals_table_3d = [[self._get_k_values(self.table[mu][n]) for n in range(len(self.table[mu]))] for mu in range(len(self.table))]
        k_distributions_table_3d = [self._get_k_distribution(k_largest_intervals_table_3d[mu]) for mu in range(len(k_largest_intervals_table_3d))]
        extremeness_table_3d = [[self._get_extremeness(k_largest_intervals_table_3d[mu][n], k_distributions_table_3d[mu]) for n in range(len(k_largest_intervals_table_3d[mu]))] for mu in range(len(k_largest_intervals_table_3d))]
        gamma_max_table_2d = [[max(extremeness_table_3d[mu][n]) for n in range(len(extremeness_table_3d[mu]))] for mu in range(len(extremeness_table_3d))]

        corresponding_cdf_values_data = self._get_corresponding_cdf_values(self.cdf)
        k_largest_intervals_data = self._get_k_values(corresponding_cdf_values_data)
        extremeness_data = [self._get_extremeness(k_largest_intervals_data, k_distributions_table_3d[mu]) for mu in range(len(k_distributions_table_3d))]
        cmaxs_data = [max(extremeness_data[mu]) for mu in range(len(extremeness_data))]
        cmaxs_data_extremeness = [self._get_extremeness(cmaxs_data[mu], gamma_max_table_2d[mu]) for mu in range(len(cmaxs_data))]
        y_filter = savgol_filter(cmaxs_data_extremeness, window_length=35, polyorder=3)

        mu_index = 0
        for i in range(len(cmaxs_data_extremeness)):
            if cmaxs_data_extremeness[i] >= self.cl:
                mu_index = i
                break
        mu_bar = self.mus[mu_index]
        sigma_bar = self.sigmas[mu_index]
        print('mu_index = ', mu_index)
        print('mu_bar = ', mu_bar)
        print('sigma_corresponding_to_cbarmax = ', sigma_bar)

        mu_index = 0
        for i in range(len(y_filter)):
            if y_filter[i] >= self.cl:
                mu_index = i
                break
        mu_bar = self.mus[mu_index]
        sigma_bar = self.sigmas[mu_index]
        print('mu_index = ', mu_index)
        print('mu_bar (filter) = ', mu_bar)
        print('sigma_corresponding_to_cbarmax = ', sigma_bar)

        plt.plot(self.mus, cmaxs_data_extremeness, label='raw')
        plt.plot(self.mus, y_filter, label='filter')
        plt.legend()
        plt.show()

        c_90percent_list = []
        c_90percent_list_2 = []
        sorted_list_gamma_max_table_mu =[]

        for mus in range(len(gamma_max_table_2d)):
            c_90percent_list.append(np.percentile(gamma_max_table_2d[mus], self.cl*100))
            sorted_list_gamma_max_table_mu = sorted(gamma_max_table_2d[mus])
            for i in range(len(sorted_list_gamma_max_table_mu)):
                if sorted_list_gamma_max_table_mu[i] >= self.cl:
                    c_90percent_list_2.append(sorted_list_gamma_max_table_mu[i])
                    break

        plt.plot(self.mus, c_90percent_list, label='90% Cmax List 1')
        plt.plot(self.mus, c_90percent_list, label='90% Cmax List 2')
        plt.xscale('log')
        plt.legend()
        plt.show()

        pass

    def make_table(self, flag: bool, number_of_lists: int, file_name: str):
        """
        Create a table for Yellin's Optimum Interval Method and writes it in a .csv file. The lengths of the lists for
        a given mu are poisson-distributed with the mean value of mu.

        :param flag: Boolean value that determines whether a new table should be created or not.
        :param number_of_lists: Number of lists wanted per mu.
        :param file_name: File path and name.
        :return: None
        """
        if flag is True:
            uniform_arrays = []
            for mu in self.mus:
                size_of_uniform_array = np.random.poisson(mu, number_of_lists)
                uniform_array = [[0.] + list(np.random.rand(size_of_uniform_array[i])) + [1.] for i in range(number_of_lists)]
                uniform_arrays.append(uniform_array)
            with open(file_name + '.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f, delimiter=' ')
                for mu in uniform_arrays:
                    for n in mu:
                        writer.writerow(n)
                    writer.writerow('\n')
            self.table = uniform_arrays
        return

    def get_data(self, file_name: str):
        """
        Read experiment data and save it into a list. All lines starting with '#' will be ignored.

        :param file_name: File path and name.
        :return: None
        """
        with open(file_name, 'r', encoding='UTF8', newline='') as f:
            dataset = f.readlines()
            dataset = [float(line.strip('\n')) for line in dataset if line[0] != '#']
        dataset.sort()
        self.data = dataset
        return

    # ------------------------------------------------
    # private functions
    # ------------------------------------------------

    @staticmethod
    def _get_k_values(energy_array: list):
        """
        Calculate k values for the given energy array. K values are the largest intervals for each k-interval.

        :param energy_array: A list of energy arrays, for which the k values are to be calculated.
        :return: K values
        :rtype: list
        """
        energy_array = np.sort(energy_array)
        k_largest_intervals_per_array = [max([energy_array[i+j]-energy_array[i] for i in range(len(energy_array)-j)])
                                         for j in range(1, len(energy_array))]
        return k_largest_intervals_per_array

    @staticmethod
    def _get_k_distribution(k_largest_intervals: list):
        """
        Make a list of all k values per k-interval which gives us the k-distribution.

        :param k_largest_intervals: K values, the largest intervals for each k-interval.
        :return: K-distribution.
        :rtype: list
        """
        df = pd.DataFrame(k_largest_intervals)
        k_distributions = [list(df[i].dropna()) for i in range(df.columns.size)]
        return k_distributions

    @staticmethod
    def _get_extremeness(k_largest_intervals_per_array, k_distributions: list):
        """
        Calculate the extremeness values.

        :param k_largest_intervals_per_array:
        :type k_largest_intervals_per_array: list or float
        :param k_distributions: K-distribution.
        :return: Extremeness value(s).
        :rtype: list or float
        """
        if type(k_largest_intervals_per_array) == list:
            number_of_common_k_intervals = min([len(k_largest_intervals_per_array), len(k_distributions)])
            extremeness = [percentileofscore(k_distributions[i], k_largest_intervals_per_array[i], kind='mean')/100.
                           for i in range(number_of_common_k_intervals)]
        elif type(k_largest_intervals_per_array) == float:
            extremeness = percentileofscore(k_distributions, k_largest_intervals_per_array, kind='mean')/100.
        else:
            raise Exception('type(k_largest_intervals_per_array) must be either list or float')
        return extremeness

    def _get_corresponding_cdf_values(self, cdf):
        """
        Calculate the corresponding cdf values to the energy values in experimental data.

        :param cdf: Cumulative density function.
        :return: Corresponding cdf values to the energy values in experimental data.
        :rtype: list
        """
        corresponding_cdf_values_to_energy = []
        for energy_value in self.data:
            for i in range(len(cdf[0])):
                if cdf[0][i] - cdf[0][0] > energy_value:
                    corresponding_cdf_values_to_energy.append(cdf[1][i-1])
                    break
        corresponding_cdf_values_to_energy = list(np.array(corresponding_cdf_values_to_energy)-min(corresponding_cdf_values_to_energy))
        corresponding_cdf_values_to_energy = list(np.array(corresponding_cdf_values_to_energy)/max(corresponding_cdf_values_to_energy))
        return corresponding_cdf_values_to_energy

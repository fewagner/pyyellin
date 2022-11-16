import csv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import percentileofscore


class Limit:
    """
    A class to calculate limits for dark matter cross-sections with Yellin's Optimum Interval Method.
    """
    def __init__(self):

        self.cl = 0.
        self.data = []
        self.corresponding_cdf = []
        self.tables_path = ''
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

    def set_confidence_level(self, cl: float):
        """
        Set confidence level.

        :param cl: Confidence level.
        :return: None
        """
        self.cl = cl
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
            with open(Path(str(Path(str(self.tables_path) + '/' + file_name)) + '.csv'), 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f, delimiter=' ')
                for mu in uniform_arrays:
                    for n in mu:
                        writer.writerow(n)
                    writer.writerow('\n')
            self.table = uniform_arrays
        return

    @staticmethod
    def _get_x_values(energy_array: list):
        """
        Calculate x values for the given energy array. X values are the largest intervals for each k-interval.

        :param energy_array: A list of energy arrays, for which the x values are to be calculated.
        :return: X values
        :rtype: list
        """
        energy_array = np.sort(energy_array)
        x_values_per_array = [max([energy_array[i+j]-energy_array[i] for i in range(len(energy_array)-j)])
                              for j in range(1, len(energy_array))]
        return x_values_per_array

    @staticmethod
    def _get_maximum_gap(energy_array: list):
        """
        Calculate maximum gap for the given energy array.

        :param energy_array: A list of energy arrays, for which the maximum gap is to be calculated.
        :return: Maximum gap.
        :rtype: float
        """
        energy_array = np.sort(energy_array)
        maximum_gap = np.max(energy_array[1:]-energy_array[:-1])
        return maximum_gap

    @staticmethod
    def _get_x_distribution(x_values: list):
        """
        Make a list of all x values per k-interval which gives us the x-distribution.

        :param x_values: X values, the largest intervals for each k-interval.
        :return: X-distribution.
        :rtype: list
        """
        df = pd.DataFrame(x_values)
        x_distributions = [list(df[i].dropna()) for i in range(df.columns.size)]
        return x_distributions

    @staticmethod
    def _get_extremeness_of_x_values(x_values_per_array, x_distributions: list):
        """
        Calculate the extremeness values of x values.

        :param x_values_per_array: X values per array.
        :param x_distributions: X-distributions.
        :return: Extremeness values.
        :rtype: list
        """
        number_of_common_k_intervals = min([len(x_values_per_array), len(x_distributions)])
        extremeness_values_per_array = [percentileofscore(x_distributions[i], x_values_per_array[i], kind='mean')/100. for i in range(number_of_common_k_intervals)]
        return extremeness_values_per_array

    @staticmethod
    def _get_extremeness_of_cmax(cmax, gamma_max_array):
        """
        Calculate the extremeness values of cmax values.

        :param cmax: Cmax value.
        :param gamma_max_array: Gamma max distribution.
        :return: Extremeness value.
        :rtype: float
        """
        extremeness_of_cmax = percentileofscore(gamma_max_array, cmax, kind='mean')/100.
        return extremeness_of_cmax

    def _get_corresponding_cdf_values(self, cdf_sum):
        """
        Calculate the corresponding cdf values to the energy values in experimental data.

        :param cdf: Cumulative density function.
        :return: Corresponding cdf values to the energy values in experimental data.
        :rtype: list
        """
        new_x = sorted(list(self.data)+list(cdf_sum[0]))
        y_interp = np.interp(new_x, cdf_sum[0], cdf_sum[1])
        corresponding_cdf_values_to_energy = y_interp[np.intersect1d(new_x, self.data, return_indices=True)[1]]
        corresponding_cdf_values_to_energy = np.sort(np.array([0] + list(corresponding_cdf_values_to_energy) + [1]))
        return corresponding_cdf_values_to_energy

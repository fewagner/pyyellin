import csv
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore
from matplotlib import pyplot as plt


class Limit:
    """
    A class to calculate Limits with Yellins Optimum Interval Method.
    """

    def __init__(self):

        self.data = []
        self.corresponding_cdf = []
        self.table = []
        self.sigmas = []
        self.mus = []
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

    def add_sigmas(self, sigmas):
        """

        :param sigmas:
        :return:
        """
        self.sigmas = sigmas
        return

    def add_mus(self, mus):
        """

        :param mus:
        :return:
        """
        self.mus = mus
        return

    def add_data(self, data: list, efficiency: list, eff_grid: list, acc_region: tuple, signal_model: object,
                 resolution: float, threshold: float, exposure: float):
        """
        Add data and definitions of a new experiment to the limit calculation.

        :param data: The recoil energies of the measured event, in keV.
        :type data: list
        :param efficiency: The binned and sorted survival probabilities of the detector.
        :type efficiency: list
        :param eff_grid: The energies corresponding to the survival probabilities, in keV.
        :type eff_grid: list
        :param acc_region: The lower and upper energy limit of the detector
        :type acc_region: tuple
        :param signal_model: A model of the expected dark matter signal for this detector.
        :type signal_model: list
        :param resolution: The resolution of the detector, in keV.
        :type resolution: list
        :param threshold: The lower threshold of the detector, in keV.
        :type threshold: list
        :param exposure: The exposure of the measurement, in kg days.
        :type exposure: list
        """

        # TODO do some asserts if the data formats are alright
        # assert statement_that_must_be_true, 'This error message is shown if the statement is not True.'

        # TODO do some assert if the Signal model is a child of the class SignalModel
        # TODO check efficiencies list is sorted
        # TODO check if eff_grid is sorted
        # TODO ...

        self.data.append(np.array(data))
        self.efficiencies.append(np.array(efficiency))
        self.eff_grids.append(np.array(eff_grid))
        self.acc_regions.append(np.array(acc_region))
        self.signal_models.append(signal_model)
        self.resolutions.append(resolution)
        self.thresholds.append(threshold)
        self.exposures.append(exposure)

        assert 1>2, "test"

    def get_limit(self, cdf):  # , signal_pars: list): TODO: cdfs as variable
        """
        Here the limit is calculated and returned.

        :param cdf:
        :param signal_pars: List of the signal parameters for all signal models for the individual experiments.
        :type signal_pars: list
        :return: The limit for the given signal parameters.
        :rtype: float
        """
        k_largest_intervals_table_3d = [[self._get_k_values(self.table[mu][n]) for n in range(len(self.table[mu]))] for mu in range(len(self.table))]
        k_distributions_table_3d = [self._get_k_distribution(k_largest_intervals_table_3d[mu]) for mu in range(len(k_largest_intervals_table_3d))]
        extremeness_table_3d = [[self._get_extremeness(k_largest_intervals_table_3d[mu][n], k_distributions_table_3d[mu]) for n in range(len(k_largest_intervals_table_3d[mu]))] for mu in range(len(k_largest_intervals_table_3d))]
        gamma_max_table_2d = [[max(extremeness_table_3d[mu][n]) for n in range(len(extremeness_table_3d[mu]))] for mu in range(len(extremeness_table_3d))]

        print('gamma_max_table_2d[-1]')
        print(sorted(gamma_max_table_2d[-1]))
        # plt.hist(gamma_max_table_2d[-1], bins=20)
        # plt.show()

        corresponding_cdf_values_data = self._get_corresponding_cdf_values(cdf)
        k_largest_intervals_data = self._get_k_values(corresponding_cdf_values_data)
        # plt.hist(self.data, bins= 100)
        # plt.hist(corresponding_cdf_values_data, bins=100)
        # plt.hist(self.table[0][0], bins=100, alpha=0.5)
        # plt.show()
        print('\nk_largest_intervals_table_3d[-1][0]')
        print(k_largest_intervals_table_3d[-1][0])
        print('\nk_largest_intervals_data')
        print(k_largest_intervals_data)
        extremeness_data = [self._get_extremeness(k_largest_intervals_data, k_distributions_table_3d[mu]) for mu in range(len(k_distributions_table_3d))]
        print('\nextremeness_data')
        print(extremeness_data)
        cmaxs_data = [max(extremeness_data[mu]) for mu in range(len(extremeness_data))]
        print('\ncmaxs_data')
        print(cmaxs_data)
        cmaxs_data_extremeness = [self._get_extremeness(cmaxs_data[mu], gamma_max_table_2d[mu]) for mu in range(len(cmaxs_data))]
        print('\ncmaxs_data_extremeness')
        print(cmaxs_data_extremeness)
        plt.plot(self.mus, cmaxs_data_extremeness)
        plt.show()
        mu_index = 0
        for i in range(len(cmaxs_data_extremeness)):
            if cmaxs_data_extremeness[i] >= 0.9:
                mu_index = i
                break
        mu_corresponding_to_cbarmax = self.mus[mu_index]
        print('mu_index = ', mu_index)
        print('mu_corresponding_to_cbarmax = ', mu_corresponding_to_cbarmax)
        # assert len(signal_pars) == len(self.table)

        # TODO call here the function for the limit calculation
        pass

    def make_table(self, flag, number_of_uniform_arrays, file_name):
        if flag is True:
            uniform_arrays = []
            for mu in self.mus:
                # size_of_uniform_array = np.array([int(mu) for i in range(number_of_uniform_arrays)])  # für gleichlange Datasets
                size_of_uniform_array = np.random.poisson(mu, number_of_uniform_arrays)
                uniform_array = [[0.] + list(np.random.rand(size_of_uniform_array[i])) + [1.] for i in range(number_of_uniform_arrays)]
                uniform_arrays.append(uniform_array)
            with open(file_name + '.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f, delimiter=' ')
                for mu in uniform_arrays:
                    for n in mu:
                        writer.writerow(n)
                    writer.writerow('\n')
        return

    def get_data(self, file_name):
        """
        This function reads experiment data and saves it into a list. All lines starting with '#' will be skipped.
        :param file_name:
        :return:
        """
        with open(file_name, 'r', encoding='UTF8', newline='') as f:
            dataset = f.readlines()
            dataset = [float(line.strip('\n')) for line in dataset if line[0] != '#']
        dataset.sort()
        self.data = dataset
        return

    def get_table(self, file_name):
        """

        :param file_name:
        :return:
        """
        dataset = []
        sub_dataset = []
        with open(file_name, 'r', encoding='UTF8', newline='') as f:
            reader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                sub_dataset.append(row)
                if row == ['\n']:
                    dataset.append(sub_dataset[:-1])
                    sub_dataset = []
        self.table = dataset
        return

    # ------------------------------------------------
    # private
    # ------------------------------------------------

    @staticmethod
    def _get_k_values(energy_array):
        """

        :param energy_array:
        :return:
        """
        energy_array = np.sort(energy_array)
        k_largest_intervals_per_array = [max([energy_array[i+j]-energy_array[i] for i in range(len(energy_array)-j)])
                                         for j in range(1, len(energy_array))]
        return k_largest_intervals_per_array

    @staticmethod
    def _get_k_distribution(k_largest_intervals):
        """

        :param k_largest_intervals:
        :return:
        """
        df = pd.DataFrame(k_largest_intervals)
        k_distributions = [list(df[i].dropna()) for i in range(df.columns.size)]
        return k_distributions

    @staticmethod
    def _get_extremeness(k_largest_intervals_per_array, k_distributions):
        """

        :param k_largest_intervals_per_array:
        :param k_distributions:
        :return:
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

    @staticmethod
    def _get_gamma_max(extremeness_per_array):
        """

        :param extremeness_per_array:
        :return:
        """
        # get max value and index of max value of an array
        gamma_max = [max(extremeness_per_array), np.where(extremeness_per_array == max(extremeness_per_array))[0][0]]
        return gamma_max

    def _get_corresponding_cdf_values(self, cdf):
        # data_sorted = np.sort(self.data)
        corresponding_cdf_values_to_energy = []
        for energy_value in self.data:
            for i in range(len(cdf[0])):
                if cdf[0][i] - cdf[0][0] > energy_value:
                    corresponding_cdf_values_to_energy.append(cdf[1][i-1])
                    break
        corresponding_cdf_values_to_energy = list(np.array(corresponding_cdf_values_to_energy)-min(corresponding_cdf_values_to_energy))
        corresponding_cdf_values_to_energy = list(np.array(corresponding_cdf_values_to_energy)/max(corresponding_cdf_values_to_energy))
        return corresponding_cdf_values_to_energy

    # here you can put function that are not callable for the user
    # these always start with an underscore:
    # def _dummy_private_func():
    #   ...
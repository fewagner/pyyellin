import numpy as np
from scipy.stats import percentileofscore


class Limit:
    """
    A class to calculate Limits with Yellins Optimum Interval Method.
    """

    def __init__(self):

        self.data = []
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
        self.eff_grid.append(np.array(eff_grid))
        self.acc_regions.append(np.array(acc_region))
        self.signal_models.append(signal_model)
        self.resolutions.append(resolution)
        self.thresholds.append(threshold)
        self.exposures.append(exposure)

        assert 1>2, "test"

    def get_limit(self, signal_pars: list):
        """
        Here the limit is calculated and returned.

        :param signal_pars: List of the signal parameters for all signal models for the individual experiments.
        :type signal_pars: list
        :return: The limit for the given signal parameters.
        :rtype: float
        """

        assert len(signal_pars) == len(self.data)

        # TODO call here the function for the limit calculation
        pass

    def tabulate(self, mus, size_of_uniform_array, number_of_uniform_arrays, max_energy):
        uniform_arrays = np.random.rand(len(mus), number_of_uniform_arrays, size_of_uniform_array)  # *max_energy size Po verteilt
        k_largest_intervals = np.array([self._get_k_values_uniform(uniform_arrays[i, j]) for j in range(number_of_uniform_arrays) for i in range(mus)])
        extremeness = np.array([self._get_extremeness(k_largest_intervals[i, j]) for j in range(number_of_uniform_arrays) for i in range(mus)])
        gamma_max_per_mu_and_N = np.array([self._get_gamma_max(extremeness[i, j]) for j in range(number_of_uniform_arrays) for i in range(mus)])
        return

    # ------------------------------------------------
    # private
    # ------------------------------------------------

    def _get_k_values_uniform(self, energy_array):
        # kmax = len(energy_array)-2  # j := in range (1, kmax+2)
        np.sort(energy_array)
        k_largest_intervals_per_array = np.array([max([energy_array[i+j]-energy_array[i]
                                                       for i in range(len(energy_array))])
                                                  for j in range(1, len(energy_array))])
        return k_largest_intervals_per_array

    def _get_extremeness(self, k_largest_intervals_per_array):
        extremeness = np.array([percentileofscore(k_largest_intervals_per_array, k_value)
                                for k_value in k_largest_intervals_per_array])/100.
        return extremeness

    def _get_gamma_max(self, extremeness_per_array):
        # get max value and index of max value of an array
        gamma_max = [max(extremeness_per_array), np.where(extremeness_per_array == max(extremeness_per_array))[0][0]]
        return gamma_max

    def _get_Cmax(self, energy_array, cdfs):  # TODO: _get_k_values anders als vorher, da wir vorher eine uniforme Verteilung hatten und jetzt es vom Signal abhängt!! CDF brauchen wir!
        np.sort(energy_array)
        mu_dimension = len(cdfs)
        kmax = len(energy_array)-2
        corresponding_cdf_values_to_energy_per_mu = []
        corresponding_cdf_values_to_energy = []
        for mu_number in range(len(cdfs)):
            for energy_value in energy_array:
                for i in range(len(cdfs[mu_number][0])):
                    if energy_value >= cdfs[mu_number][0][i]:
                        corresponding_cdf_values_to_energy_per_mu.append(cdfs[mu_number][1][i])
                        break
            corresponding_cdf_values_to_energy.append(corresponding_cdf_values_to_energy_per_mu)
        corresponding_cdf_values_to_energy = np.array(corresponding_cdf_values_to_energy)
        k_largest_intervals = np.array([self._get_k_values_uniform(corresponding_cdf) for corresponding_cdf in corresponding_cdf_values_to_energy])
        c_values_per_mu = np.array([self._get_extremeness(intervals_per_mu)] for intervals_per_mu in k_largest_intervals)
        c_max_values_per_mu = np.array([self._get_gamma_max(c_values) for c_values in c_values_per_mu])
        return c_max_values_per_mu

    def _get_Cmax2(self, energy_array, cdfs):  # TODO: _get_k_values anders als vorher, da wir vorher eine uniforme Verteilung hatten und jetzt es vom Signal abhängt!! CDF brauchen wir!
        np.sort(energy_array)
        mu_dimension = len(cdfs)
        kmax = len(energy_array)-2
        corresponding_cdf_values_to_energy_per_mu = []
        corresponding_cdf_values_to_energy = []
        for mu_number in range(len(cdfs)):
            for energy_value in energy_array:
                for i in range(len(cdfs[mu_number][0])):
                    if energy_value >= cdfs[mu_number][0][i]:
                        corresponding_cdf_values_to_energy_per_mu.append(cdfs[mu_number][1][i])
                        break
            corresponding_cdf_values_to_energy.append(corresponding_cdf_values_to_energy_per_mu)
        corresponding_cdf_values_to_energy = np.array(corresponding_cdf_values_to_energy)
        k_largest_intervals = np.array([max([corresponding_cdf_values_to_energy[mu_number][i+j]-corresponding_cdf_values_to_energy[mu_number][i]
                                             for i in range(len(energy_array))])
                                        for j in range(1, len(energy_array))
                                        for mu_number in range(mu_dimension)])
        c_values_per_mu = np.array([percentileofscore(k_largest_intervals[mu_number], k_value)
                                    for k_value in k_largest_intervals[mu_number]
                                    for mu_number in range(mu_dimension)])/100.
        c_max_values_per_mu = np.array([max(c_values_per_mu[mu_number]) for mu_number in range(mu_dimension)])
        return c_max_values_per_mu

    # here you can put function that are not callable for the user
    # these always start with an underscore:
    # def _dummy_private_func():
    #   ...
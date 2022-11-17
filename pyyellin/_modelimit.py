import json
import numpy as np
import pyyellin as yell
from pathlib import Path
from os.path import exists
from scipy.signal import savgol_filter


class ModeLimit(yell.SignalModel, yell.Limit):
    """
    The class combining signal model and limit classes, hence the name ModeLimit, that can calculate everything from
    SignalModel and from Limit and in addition can calculate upper limits using either Yellin's maximum gap method or
    optimum interval method.
    """

    def __init__(self):
        super().__init__()
        self.size = 10
        self.masses = [10.]
        self.sigmas = []
        self.mus = []
        self.mus_from_cdf = []
        self.mus_from_other_model = []
        self.cdf = []
        self.table = []
        self.c_90percent_list = []
        self.c_90_savgol = []
        self.masses_for_plot_oi_sm = []
        self.mus_corresponding_to_cbarmax_list_oi_sm = []
        self.sigmas_corresponding_to_cbarmax_list_oi_sm = []
        self.masses_for_plot_mg_sm = []
        self.mus_corresponding_to_cbarmax_list_mg_sm = []
        self.sigmas_corresponding_to_cbarmax_list_mg_sm = []
        self.masses_for_plot_oi_am = []
        self.mus_corresponding_to_cbarmax_list_oi_am = []
        self.sigmas_corresponding_to_cbarmax_list_oi_am = []
        self.masses_for_plot_mg_am = []
        self.mus_corresponding_to_cbarmax_list_mg_am = []
        self.sigmas_corresponding_to_cbarmax_list_mg_am = []
        self.make_table_bool = True
        self.number_of_lists = 100
        self.results_path = ''
        self.table_file_name = 'table'

    def set_masses(self, masses):
        """
        Set dark matter particle masses.

        :param masses: Dark matter particle masses.
        :type masses: list or ndarray
        :return:
        """
        self.masses = masses
        return

    def set_table_and_results_paths(self, tables_path, results_path):
        """
        Set the paths for the tables and results.

        :param tables_path: Path for the tables.
        :param results_path: Path for the results.
        :return:
        """
        self.tables_path = Path(tables_path)
        self.results_path = Path(results_path)
        return

    def set_table_variables(self, flag, number_of_lists, file_name):
        """
        Set required variables for creating the table needed for the Yellin's Optimum Interval Method.

        :param flag: Boolean value that determines whether a new table should be created or not.
        :param number_of_lists: Number of lists wanted per mu.
        :param file_name: File path and name.
        :return: None
        """
        self.make_table_bool = flag
        self.number_of_lists = number_of_lists
        self.table_file_name = file_name
        if flag is True:
            value = input('Are you sure that you want to make a new table?(y/n)\n')
            if value == 'n':
                exit()
        else:
            value = input('Are you sure that you do not want to make a new table?(y/n)\n')
            if value == 'n':
                exit()
        return

    def set_sigma_interval(self, min_sigma, max_sigma):
        """
        Set min and max values of the cross-section interval. This function is to be used when our model is used for
        the calculation of pdfs, cdfs, mus.

        :param min_sigma: Minimum cross-section.
        :param max_sigma: Maximum cross-section.
        :return: None
        """
        self.sigmas = np.array([min_sigma, max_sigma])
        return

    def set_sigmas(self, min_sigma, max_sigma, size):
        """
        Set min and max values of the cross-section interval for maximum gap analysis. This function is to be used when our model is used for
        the calculation of pdfs, cdfs, mus.

        :param min_sigma: Minimum cross-section.
        :param max_sigma: Maximum cross-section.
        :param size: Array size.
        :return: None
        """
        self.sigmas = np.geomspace(min_sigma, max_sigma, size)
        return

    def set_mu_interval(self, min_mu, max_mu, size):
        """
        Set min and max values of the interval for expected number of events (μ).

        :param min_mu: Minimum expected number of events (μ). Minimum should be around Max(mus).
        :param max_mu: Maximum expected number of events (μ). Maximum should be around Max(mus).
        :param size: Size of the array.
        :return: None
        """
        self.mus = np.geomspace(min_mu, max_mu, size)
        return

    @staticmethod
    def calculate_c_0(x, mu):
        """
        Probability of the maximum gap size being smaller than a particular value of x.
        :param mu: The expected value of recoils in the gap.
        :param x: The value with that we compare the maximum gap size.
        """
        if mu < 0.:
            if x < 1.:
                return 0.
            else:
                return 1.

        elif x > mu:
            return 1.

        elif (x < .03 * mu and mu < 60.) \
                or (x < 1.8 and mu >= 60.) \
                or (x < 4. and mu > 700.):
            return 0.

        else:
            m = int(mu/x)
            exp_term = np.exp(-x)
            if m >= 1:
                c_0 = 1.
                factorial_term = 1.
                for k in range(1, m+1):
                    factorial_term /= k
                    parenthesis_term = k * x - mu
                    summand_term = exp_term ** k * parenthesis_term ** (k-1) * factorial_term * (parenthesis_term - k)
                    c_0 += summand_term
                    if np.abs(summand_term) < 1e-10:
                        break
            else:
                return 1
        return c_0

    def get_limit_maximum_gap(self):
        """
        Calculate the limit for the cross-section of dark matter particles using Yellin's maximum gap method.

        :return: None
        """
        self.data = np.array(self.data)
        self.data = self.data[(self.data > self.threshold) & (self.data < self.upper_integral_limit)]

        for mass in self.masses:
            print('Loop number =', list(self.masses).index(mass) + 1, '/', len(self.masses))
            print(f'mass = {mass} GeV')
            pdf = self.pdf(mass)
            pdf_sum = self.pdf_sum_from_pdf(mass, pdf, self.materials)
            cdf_sum = self.cdf_sum_from_pdf_sum(pdf_sum)
            self.cdf = np.copy(cdf_sum)
            mus = self.get_mus(mass, pdf_sum, self.sigmas)
            self.mus_from_cdf = np.copy(mus)

            corresponding_cdf_values_data = self._get_corresponding_cdf_values(self.cdf)
            maximum_gap = self._get_maximum_gap(corresponding_cdf_values_data)
            maximum_gap *= mus
            c_zeros = []
            for mu, max_gap in zip(mus, maximum_gap):
                c_zeros.append(self.calculate_c_0(max_gap, mu))
            c_zeros = np.array(c_zeros)
            if (len(c_zeros[c_zeros > self.cl]) != 0) and (len(c_zeros[c_zeros < self.cl]) != 0):
                self.masses_for_plot_mg_sm.append(mass)
                index = np.where(c_zeros > self.cl)[0][0]
                self.mus_corresponding_to_cbarmax_list_mg_sm.append(self.mus_from_cdf[index])
                self.sigmas_corresponding_to_cbarmax_list_mg_sm.append(self.sigmas[index])

        with open(Path(str(self.results_path) + '/' + 'maximum_gap_limits_sm.dat'), 'w') as dat_file:
            dat_file.write("# Calculated limits using Yellin's optimum interval method.\n# Mass in GeV (left), cross"
                           "sections in pb (center), expected number of events (right).\n")
            for mass, sigma, mu in zip(self.masses_for_plot_mg_sm, self.sigmas_corresponding_to_cbarmax_list_mg_sm, self.mus_corresponding_to_cbarmax_list_mg_sm):
                dat_file.write(f'{mass}  {sigma}  {mu}\n')
        return

    def get_limit(self):
        """
        Calculate the limit for the cross-section of dark matter particles using Yellin's optimum interval method.

        :return: None
        """
        self.data = np.array(self.data)
        self.data = self.data[(self.data > self.threshold) & (self.data < self.upper_integral_limit)]

        if exists(Path(str(self.tables_path) + '/' + f'x_distributions_table_3d_{self.table_file_name}.json')) and \
                exists(Path(str(self.tables_path) + '/' + f'gamma_max_table_2d_{self.table_file_name}.json')) and \
                (not self.make_table_bool) is True:
            x_distributions_table_3d = json.load(open(Path(str(self.tables_path) + '/' + f'x_distributions_table_3d_{self.table_file_name}.json')))
            gamma_max_table_2d = json.load(open(Path(str(self.tables_path) + '/' + f'gamma_max_table_2d_{self.table_file_name}.json')))
        else:
            value = input('Table seems to be missing. Do you want to make one?(y/n)\n')
            if value == 'n':
                exit()
            self.make_table(self.make_table_bool, self.number_of_lists, self.table_file_name)
            x_values_table_3d = [[self._get_x_values(self.table[mu_index][n]) for n in range(len(self.table[mu_index]))] for mu_index in range(len(self.table))]
            x_distributions_table_3d = [self._get_x_distribution(x_values_table_3d[mu_index]) for mu_index in range(len(x_values_table_3d))]
            extremeness_table_3d = [[self._get_extremeness_of_x_values(x_values_table_3d[mu_index][n], x_distributions_table_3d[mu_index]) for n in range(len(x_values_table_3d[mu_index]))] for mu_index in range(len(x_values_table_3d))]
            gamma_max_table_2d = [[max(extremeness_table_3d[mu_index][n]) for n in range(len(extremeness_table_3d[mu_index]))] for mu_index in range(len(extremeness_table_3d))]
            with open(Path(str(self.tables_path) + '/' + f'x_distributions_table_3d_{self.table_file_name}.json'), "w") as f:
                f.write(json.dumps(x_distributions_table_3d))
            with open(Path(str(self.tables_path) + '/' + f'gamma_max_table_2d_{self.table_file_name}.json'), "w") as f:
                f.write(json.dumps(gamma_max_table_2d))

        # self.c_90percent_list = []  # TODO: Frag nach, nur für die Projektarbeit/Präsentation wichtig oder?
        # for mu_index in range(len(gamma_max_table_2d)):
        #     self.c_90percent_list.append(np.percentile(gamma_max_table_2d[mu_index], self.cl*100))
        # self.c_90_savgol = savgol_filter(self.c_90percent_list, window_length=25, polyorder=3)

        self.masses_for_plot_oi_sm = []
        for mass in self.masses:
            print('Loop number =', list(self.masses).index(mass) + 1, '/', len(self.masses))
            print(f'mass = {mass} GeV')
            pdf = self.pdf(mass)
            pdf_sum = self.pdf_sum_from_pdf(mass, pdf, self.materials)
            cdf_sum = self.cdf_sum_from_pdf_sum(pdf_sum)
            self.cdf = np.copy(cdf_sum)
            mus = self.get_mus(mass, pdf_sum, self.sigmas)
            self.mus_from_cdf = np.copy(mus)

            corresponding_cdf_values_data = self._get_corresponding_cdf_values(self.cdf)
            x_values_data = self._get_x_values(corresponding_cdf_values_data)
            extremeness_data = [self._get_extremeness_of_x_values(x_values_data, x_distributions_table_3d[mu_index]) for mu_index in range(len(x_distributions_table_3d))]
            cmaxs_data = [max(extremeness_data[mu]) for mu in range(len(extremeness_data))]
            cmaxs_data_extremeness = [self._get_extremeness_of_cmax(cmaxs_data[mu_index], gamma_max_table_2d[mu_index]) for mu_index in range(len(cmaxs_data))]

            above_count, below_count = 0, 0  # tODO do this part with y_filter and enough if it crosses 0.9
            there_are_values_above_90, there_are_values_below_90 = False, False
            for i in range(len(cmaxs_data_extremeness)):
                if cmaxs_data_extremeness[i] >= self.cl:
                    above_count += 1
                    if above_count == int(0.05*len(self.mus)):
                        there_are_values_above_90 = True
                else:
                    below_count += 1
                    if below_count == int(0.05*len(self.mus)):
                        there_are_values_below_90 = True
            if there_are_values_above_90 == there_are_values_below_90 == True:
                print('Acceptable mu_bar can be found for this mass in this mu interval.')
            elif there_are_values_above_90 is True and there_are_values_below_90 is False:
                print('This mu interval is not suited for this mass, hence all cmax values seem to be above 90%.')
            elif there_are_values_above_90 is False and there_are_values_below_90 is True:
                print('This mu interval is not suited for this mass, hence all cmax values seem to be below 90%.')
            else:
                print('There seems to be a problem with the calculation of cmax values.')

            if there_are_values_above_90 == there_are_values_below_90 == True:

                y_filter_cmaxs_data_extremeness = savgol_filter(cmaxs_data_extremeness, window_length=35, polyorder=3)
                mu_bar = np.nan
                for mu_index in range(len(self.mus)):
                    if y_filter_cmaxs_data_extremeness[mu_index] >= np.percentile(gamma_max_table_2d[mu_index], self.cl*100):
                        print(y_filter_cmaxs_data_extremeness[mu_index], np.percentile(gamma_max_table_2d[mu_index], self.cl*100))
                        mu_bar = self.mus[mu_index]
                        break
                sigma_bar = self._find_sigma_bar_from_mu_linear_model(mu_bar)
                print('mu_bar = ', mu_bar)
                print('sigma_bar = ', sigma_bar)

                self.masses_for_plot_oi_sm.append(mass)
                self.mus_corresponding_to_cbarmax_list_oi_sm.append(mu_bar)
                self.sigmas_corresponding_to_cbarmax_list_oi_sm.append(sigma_bar)
        with open(Path(str(self.results_path) + '/' + 'optimum_interval_limits_sm.dat'), 'w') as dat_file:
            dat_file.write("# Calculated limits using Yellin's optimum interval method.\n# Mass in GeV (left), cross"
                           "sections in pb (center), expected number of events (right).\n")
            for mass, sigma, mu in zip(self.masses_for_plot_oi_sm, self.sigmas_corresponding_to_cbarmax_list_oi_sm, self.mus_corresponding_to_cbarmax_list_oi_sm):
                dat_file.write(f'{mass}  {sigma}  {mu}\n')
        return

    def _find_sigma_bar_from_mu_linear_model(self, mu_bar):
        """
        Derive the corresponding cross-section (σ) from the expected number of events (µ). These two are linearly
        dependent.
        :param mu_bar:  Mu, whereat the Cmax is Cbarmax(confidence_level)
        :return: sigma_bar
        """
        slope_of_line = (self.mus_from_cdf[-1]-self.mus_from_cdf[0])/(self.sigmas[-1]-self.sigmas[0])
        intercept_of_line = self.mus_from_cdf[-1]-slope_of_line*self.sigmas[-1]
        sigma_bar = (mu_bar-intercept_of_line)/slope_of_line
        return sigma_bar

    #### Additional Functions ####

    def add_cdf(self, cdf):
        """
        Add cumulative density function.

        :param cdf: Cumulative density function.
        :return: None
        """
        self.cdf = cdf
        return

    def set_sampling_size(self, size: int):
        """
        Set sampling size for the rvs function.

        :param size: Size for the rvs function.
        :return: None
        """
        self.size = size
        return

    def add_sigmas_and_mus(self, sigmas, mus):
        """
        Add sigmas and corresponding mus. This function is to be used when another model is used for the calculation of
        pdfs, cdfs and mus.

        :param sigmas: Cross-sections.
        :param mus: Expected number of events per mass.
        :return: None
        """
        self.sigmas = sigmas
        self.mus_from_other_model = mus
        return

    def _find_sigma_bar_from_mu_nonlinear_model(self, mu_bar):
        """
        Derive the corresponding cross-section (σ) from the expected number of events (µ). These two are nonlinearly
        dependent, hence a linear interpolation function will be used to derive the σ.
        :param mu_bar:  Mu, whereat the Cmax is Cbarmax(confidence_level)
        :return: sigma_bar
        """
        new_mus_list = list(np.copy(self.mus_from_cdf))
        new_mus_list.append(mu_bar)
        new_mus_list = sorted(new_mus_list)
        sigma_interp_list = np.interp(new_mus_list, self.mus_from_cdf, self.sigmas)
        sigma_bar = sigma_interp_list[new_mus_list.index(mu_bar)]
        return sigma_bar

    def calculate_items(self, pdf_bool: bool = True, cdf_bool: bool = True, samples_bool: bool = False,
                        pdf_sum_bool: bool = True, cdf_sum_bool: bool = True, samples_sum_bool: bool = False,
                        mus_bool: bool = True):
        """
        Calculate the following based on the WIMP model for dark matter.

        :param pdf_bool: Probability density functions.
        :param cdf_bool: Cumulative density functions.
        :param samples_bool: Random variable samples.
        :param pdf_sum_bool: Sum the pdfs of the given materials according to their percentage in the molecule.
        :param cdf_sum_bool: Sum the cdfs of the given materials according to their percentage in the molecule.
        :param samples_sum_bool: Random variable samples using cdf_sum.
        :param mus_bool: Expected number of events based on pdf_sum_convoluted.
        :return: List of wanted parameters in the following order: pdf, cdf, samples, pdf_sum, cdf_sum, samples_sum,
        pdf_sum_convoluted, cdf_sum_convoluted, samples_sum_convoluted, mus.
        :rtype: list
        """
        if cdf_bool is True and pdf_bool is False:
            raise Exception('Set pdf_bool True in order to calculate cdf.')
        if samples_bool is True and (pdf_bool or cdf_bool) is False:
            raise Exception('Set pdf_bool and cdf_bool True in order to calculate samples.')
        if pdf_sum_bool is True and pdf_bool is False:
            raise Exception('Set pdf_bool True in order to calculate pdf_sum.')
        if cdf_sum_bool is True and (pdf_bool or pdf_sum_bool) is False:
            raise Exception('Set pdf_bool and pdf_sum_bool True in order to calculate cdf_sum.')
        if samples_sum_bool is True and (pdf_bool or pdf_sum_bool or cdf_sum_bool) is False:
            raise Exception('Set pdf_bool, pdf_sum_bool and cdf_sum_bool True in order to calculate samples_sum.')
        if mus_bool is True and (pdf_bool or pdf_sum_bool) is False:
            raise Exception('Set pdf_bool and pdf_sum True in order to calculate mus.')

        what_to_return_dict = {}
        for mass in self.masses:
            if pdf_bool is True:
                pdf = self.pdf(mass)
            else:
                pdf = None
            if cdf_bool is True:
                cdf = self.cdf_from_pdf(pdf)
            else:
                cdf = None
            if samples_bool is True:
                samples = self.rvs_from_cdf(self.size, cdf)
            else:
                samples = None
            if pdf_sum_bool is True:
                pdf_sum = self.pdf_sum_from_pdf(mass, pdf, self.materials)
            else:
                pdf_sum = None
            if cdf_sum_bool is True:
                cdf_sum = self.cdf_sum_from_pdf_sum(pdf_sum)
            else:
                cdf_sum = None
            if samples_sum_bool is True:
                samples_sum = self.rvs_sum_from_cdf_sum(self.size, cdf_sum)
            else:
                samples_sum = None
            if mus_bool is True:
                mus = self.get_mus(mass, pdf_sum, self.sigmas)
            else:
                mus = None
            what_to_return = {}
            for name, element in zip(['pdf', 'cdf', 'samples', 'pdf_sum', 'cdf_sum', 'samples_sum', 'mus'], [pdf, cdf, samples, pdf_sum, cdf_sum, samples_sum, mus]):
                what_to_return[name] = element
            what_to_return_dict[mass] = what_to_return
        return what_to_return_dict

    def get_limit_from_other_model_maximum_gap(self):
        """
        Calculate the limit for the cross-section of dark matter particles using Yellin's maximum gap method.

        :return: None
        """
        self.data = np.array(self.data)
        self.data = self.data[(self.data > self.threshold) & (self.data < self.upper_integral_limit)]

        for mass in self.masses:
            print('Loop number =', list(self.masses).index(mass) + 1, '/', len(self.masses))
            print(f'mass = {mass} GeV')
            cdf_for_this_mass = self.cdf[list(self.masses).index(mass)]
            mus = self.mus_from_other_model[list(self.masses).index(mass)]

            corresponding_cdf_values_data = self._get_corresponding_cdf_values(cdf_for_this_mass)
            maximum_gap = self._get_maximum_gap(corresponding_cdf_values_data)
            maximum_gap *= mus  # tODO CHECK THIS
            c_zeros = []
            for mu, max_gap in zip(mus, maximum_gap):
                c_zeros.append(self.calculate_c_0(max_gap, mu))
            c_zeros = np.array(c_zeros)
            if (len(c_zeros[c_zeros > self.cl]) != 0) and (len(c_zeros[c_zeros < self.cl]) != 0):
                self.masses_for_plot_mg_am.append(mass)
                index = np.where(c_zeros > self.cl)[0][0]
                self.mus_corresponding_to_cbarmax_list_mg_am.append(self.mus_from_cdf[index])
                self.sigmas_corresponding_to_cbarmax_list_mg_am.append(self.sigmas[index])

        with open(Path(str(self.results_path) + '/' + 'maximum_gap_limits_am.dat'), 'w') as dat_file:
            dat_file.write("# Calculated limits using Yellin's optimum interval method.\n# Mass in GeV (left), cross"
                           "sections in pb (center), expected number of events (right).\n")
            for mass, sigma, mu in zip(self.masses_for_plot_mg_am, self.sigmas_corresponding_to_cbarmax_list_mg_am, self.mus_corresponding_to_cbarmax_list_mg_am):
                dat_file.write(f'{mass}  {sigma}  {mu}\n')
        return

    def get_limit_from_other_model(self, dependency_is_linear: bool):
        """
        Calculate the limit for the cross-section of dark matter particles using Yellin's optimum interval method.

        :param dependency_is_linear: Whether the dependancy of mus on sigmas is linear or not.
        :return: None
        """
        self.data = np.array(self.data)
        self.data = self.data[(self.data > self.threshold) & (self.data < self.upper_integral_limit)]

        if exists(Path(str(self.tables_path) + '/' + f'x_distributions_table_3d_{self.table_file_name}.json')) and \
                exists(Path(str(self.tables_path) + '/' + f'gamma_max_table_2d_{self.table_file_name}.json')) and \
                (not self.make_table_bool) is True:
            x_distributions_table_3d = json.load(open(Path(str(self.tables_path) + '/' + f'x_distributions_table_3d_{self.table_file_name}.json')))
            gamma_max_table_2d = json.load(open(Path(str(self.tables_path) + '/' + f'gamma_max_table_2d_{self.table_file_name}.json')))
        else:
            value = input('Table seems to be missing. Do you want to make one?(y/n)\n')
            if value == 'n':
                exit()
            self.make_table(self.make_table_bool, self.number_of_lists, self.table_file_name)
            x_values_table_3d = [[self._get_x_values(self.table[mu_index][n]) for n in range(len(self.table[mu_index]))] for mu_index in range(len(self.table))]
            x_distributions_table_3d = [self._get_x_distribution(x_values_table_3d[mu_index]) for mu_index in range(len(x_values_table_3d))]
            extremeness_table_3d = [[self._get_extremeness_of_x_values(x_values_table_3d[mu_index][n], x_distributions_table_3d[mu_index]) for n in range(len(x_values_table_3d[mu_index]))] for mu_index in range(len(x_values_table_3d))]
            gamma_max_table_2d = [[max(extremeness_table_3d[mu_index][n]) for n in range(len(extremeness_table_3d[mu_index]))] for mu_index in range(len(extremeness_table_3d))]
            with open(Path(str(self.tables_path) + '/' + f'x_distributions_table_3d_{self.table_file_name}.json'), "w") as f:
                f.write(json.dumps(x_distributions_table_3d))
            with open(Path(str(self.tables_path) + '/' + f'gamma_max_table_2d_{self.table_file_name}.json'), "w") as f:
                f.write(json.dumps(gamma_max_table_2d))

        # self.c_90percent_list = []  # TODO: ??
        # for mu_index in range(len(gamma_max_table_2d)):
        #     self.c_90percent_list.append(np.percentile(gamma_max_table_2d[mu_index], self.cl*100))
        # self.c_90_savgol = savgol_filter(self.c_90percent_list, window_length=25, polyorder=3)

        self.masses_for_plot_oi_am = []
        for mass in self.masses:
            print('Loop number =', list(self.masses).index(mass) + 1)
            print('mass =', mass)
            cdf_for_this_mass = self.cdf[list(self.masses).index(mass)]
            self.mus_from_cdf = self.mus_from_other_model[list(self.masses).index(mass)]

            corresponding_cdf_values_data = self._get_corresponding_cdf_values(cdf_for_this_mass)
            x_values_data = self._get_x_values(corresponding_cdf_values_data)
            extremeness_data = [self._get_extremeness_of_x_values(x_values_data, x_distributions_table_3d[mu_index]) for mu_index in range(len(x_distributions_table_3d))]
            cmaxs_data = [max(extremeness_data[mu]) for mu in range(len(extremeness_data))]
            cmaxs_data_extremeness = [self._get_extremeness_of_cmax(cmaxs_data[mu_index], gamma_max_table_2d[mu_index]) for mu_index in range(len(cmaxs_data))]

            y_filter = savgol_filter(cmaxs_data_extremeness, window_length=35, polyorder=3)

            above_count, below_count = 0, 0
            there_are_values_above_90, there_are_values_below_90 = False, False
            for i in range(len(cmaxs_data_extremeness)):
                if cmaxs_data_extremeness[i] >= self.cl:
                    above_count += 1
                    if above_count == int(0.05*len(self.mus)):
                        there_are_values_above_90 = True
                else:
                    below_count += 1
                    if below_count == int(0.05*len(self.mus)):
                        there_are_values_below_90 = True
            if there_are_values_above_90 == there_are_values_below_90 == True:
                print('Acceptable mu_bar can be found for this mass in this mu interval.')
            elif there_are_values_above_90 is True and there_are_values_below_90 is False:
                print('This mu interval is not suited for this mass, hence all cmax values seem to be above 90%.')
            elif there_are_values_above_90 is False and there_are_values_below_90 is True:
                print('This mu interval is not suited for this mass, hence all cmax values seem to be below 90%.')
            else:
                print('There seems to be a problem with the calculation of cmax values.')

            if there_are_values_above_90 == there_are_values_below_90 == True:

                y_filter_cmaxs_data_extremeness = savgol_filter(cmaxs_data_extremeness, window_length=35, polyorder=3)
                mu_bar = np.nan
                for mu_index in range(len(self.mus)):
                    if y_filter_cmaxs_data_extremeness[mu_index] >= np.percentile(gamma_max_table_2d[mu_index], self.cl*100):
                        print(y_filter_cmaxs_data_extremeness[mu_index], np.percentile(gamma_max_table_2d[mu_index], self.cl*100))
                        mu_bar = self.mus[mu_index]
                        break
                if dependency_is_linear is True:
                    sigma_bar = self._find_sigma_bar_from_mu_linear_model(mu_bar)
                else:
                    sigma_bar = self._find_sigma_bar_from_mu_nonlinear_model(mu_bar)
                print('mu_bar = ', mu_bar)
                print('sigma_bar = ', sigma_bar)

                self.masses_for_plot_oi_am.append(mass)
                self.mus_corresponding_to_cbarmax_list_oi_am.append(mu_bar)
                self.sigmas_corresponding_to_cbarmax_list_oi_am.append(sigma_bar)
        if dependency_is_linear is True:
            str_identifier = 'linear'
        else:
            str_identifier = 'nonlinear'
        with open(Path(str(self.results_path) + '/' + f'optimum_interval_limits_am_{str_identifier}.dat'), 'w') as dat_file:
            dat_file.write("# Calculated limits using Yellin's optimum interval method.\n# Mass in GeV (left), cross"
                           "sections in pb (center), expected number of events (right).\n")
            for mass, sigma, mu in zip(self.masses_for_plot_oi_am, self.sigmas_corresponding_to_cbarmax_list_oi_am, self.mus_corresponding_to_cbarmax_list_oi_am):
                dat_file.write(f'{mass}  {sigma}  {mu}\n')
        return

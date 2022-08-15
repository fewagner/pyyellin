import numpy as np
import pyyellin as yell
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from os.path import exists
import json


class Ilok(yell.SignalModel, yell.Limit):
    """
    The ultimate chad class that can calculate everything from SignalModel and from Limit.
    """

    def __init__(self):

        super().__init__()
        self.size = 10
        self.masses = [10.]
        self.sigmas = []
        self.mus = []
        self.mus_from_cdf = []
        self.cdf = []
        self.table = []
        self.mus_corresponding_to_cbarmax_list = []
        self.sigmas_corresponding_to_cbarmax_list = []
        self.make_table_bool = True
        self.number_of_lists = 100
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

    def set_sampling_size(self, size: int):
        """
        Set sampling size for the rvs function.

        :param size: Size for the rvs function.
        :return: None
        """
        self.size = size
        return

    def set_sigmas(self, sigmas):  # only if our model is used, otherwise users may choose to add their own sigmas & mus
        """
        Set cross-sections. This function is to be used when our model is used for the calculation of pdfs, cdfs, mus.

        :param sigmas: Cross-sections.
        :type sigmas: list or ndarray
        :return:
        """
        self.sigmas = sigmas
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

    def set_mu_interval(self, min_mu, max_mu):
        """
        Set min and max values of the interval for expected number of events (μ).

        :param min_mu: Minimum expected number of events (μ). Minimum should be around Max(mus).
        :param max_mu: Maximum expected number of events (μ). Maximum should be around Max(mus).
        :return: None
        """
        self.mus = np.geomspace(min_mu, max_mu, 1000)
        # print('real mus =', self.mus)
        return

    def add_sigmas_and_mus(self, sigmas, mus):
        """
        Add sigmas and corresponding mus. This function is to be used when another model is used for the calculation of
        pdfs, cdfs and mus.

        :param sigmas: Cross-sections.
        :param mus: Expected number of events.
        :return: None
        """
        self.sigmas = sigmas
        self.mus_from_cdf = mus
        return

    def calculate_items(self, pdf_bool: bool = True, cdf_bool: bool = True, samples_bool: bool = False,
                        pdf_sum_bool: bool = True, cdf_sum_bool: bool = True, samples_sum_bool: bool = False,
                        pdf_sum_convoluted_bool: bool = True, cdf_sum_convoluted_bool: bool = True, samples_sum_convoluted_bool: bool = False,
                        mus_bool: bool = True):
        """
        Calculate the following based on the WIMP model for dark matter.

        :param pdf_bool: Probability density functions.
        :param cdf_bool: Cumulative density functions.
        :param samples_bool: Random variable samples.
        :param pdf_sum_bool: Sum the pdfs of the given materials according to their percentage in the molecule.
        :param cdf_sum_bool: Sum the cdfs of the given materials according to their percentage in the molecule.
        :param samples_sum_bool: Random variable samples using cdf_sum.
        :param pdf_sum_convoluted_bool: The convoluted and summed pdf in order to take into account the finite energy resolution.
        :param cdf_sum_convoluted_bool: The convoluted and summed cdf in order to take into account the finite energy resolution.
        :param samples_sum_convoluted_bool: Random variable samples using cdf_sum_convoluted.
        :param mus_bool: Expected number of events based on pdf_sum_convoluted.
        :return: List of wanted parameters in the following order: pdf, cdf, samples, pdf_sum, cdf_sum, samples_sum,
        pdf_sum_convoluted, cdf_sum_convoluted, samples_sum_convoluted, mus.
        :rtype: list
        """
        if cdf_bool is True and pdf_bool is False:
            raise Exception('Set pdf True in order to calculate cdf.')
        if samples_bool is True and (pdf_bool or cdf_bool) is False:
            raise Exception('Set pdf and cdf True in order to calculate samples.')
        if pdf_sum_bool is True and pdf_bool is False:
            raise Exception('Set pdf True in order to calculate pdf_sum.')
        if cdf_sum_bool is True and (pdf_bool or cdf_bool) is False:
            raise Exception('Set pdf and cdf True in order to calculate cdf_sum.')
        if samples_sum_bool is True and (pdf_bool or cdf_bool or cdf_sum_bool) is False:
            raise Exception('Set pdf, cdf and cdf_sum True in order to calculate samples_sum.')
        if pdf_sum_convoluted_bool is True and (pdf_bool or pdf_sum_bool) is False:
            raise Exception('Set pdf, pdf_sum True in order to calculate pdf_sum_convoluted.')
        if cdf_sum_convoluted_bool is True and (pdf_bool or pdf_sum_bool or pdf_sum_convoluted_bool) is False:
            raise Exception('Set pdf, pdf_sum and pdf_sum_convoluted True in order to calculate cdf_sum_convoluted.')
        if samples_sum_convoluted_bool is True and (pdf_bool or pdf_sum_bool or pdf_sum_convoluted_bool or cdf_sum_convoluted_bool) is False:
            raise Exception('Set pdf, pdf_sum, pdf_sum_convoluted and cdf_sum_convoluted True in order to calculate samples_sum_convoluted.')
        if mus_bool is True and (pdf_bool or pdf_sum_bool or pdf_sum_convoluted_bool) is False:
            raise Exception('Set pdf, pdf_sum and pdf_sum_convoluted True in order to calculate mus.')

        what_to_return_lists = []
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
                samples = self.rvs2(self.size, cdf)
            else:
                samples = None
            if pdf_sum_bool is True:
                pdf_sum = self.pdf_sum(pdf, self.materials)
            else:
                pdf_sum = None
            if cdf_sum_bool is True:
                cdf_sum = self.cdf_sum(cdf, self.materials)
            else:
                cdf_sum = None
            if samples_sum_bool is True:
                samples_sum = self.rvs_sum(self.size, cdf_sum)
            else:
                samples_sum = None
            if pdf_sum_convoluted_bool is True:
                pdf_sum_convoluted = self.pdf_sum_convolution(pdf_sum)
            else:
                pdf_sum_convoluted = None
            if cdf_sum_convoluted_bool is True:
                cdf_sum_convoluted = self.cdf_from_convoluted_pdf(pdf_sum_convoluted)
            else:
                cdf_sum_convoluted = None
            if samples_sum_convoluted_bool is True:
                samples_sum_convoluted = self.rvs_sum(self.size, cdf_sum_convoluted)
            else:
                samples_sum_convoluted = None
            if mus_bool is True:  # TODO: get mus with pdf_sum_convoluted if True, otherwise with pdf_sum
                mus = self.get_mus(pdf_sum_convoluted, self.sigmas)
            else:
                mus = None
            what_to_return = []
            for element in [pdf, cdf, samples, pdf_sum, cdf_sum, samples_sum, pdf_sum_convoluted, cdf_sum_convoluted,
                            samples_sum_convoluted, mus]:
                what_to_return.append(element)
            what_to_return_lists.append(what_to_return)
        return what_to_return_lists

    def get_limit(self):
        """
        Calculate the limit for the cross-section of dark matter particles using Yellin's Optimum Interval Method.

        :return: None
        """
        self.data = np.array([element for element in self.data if element >= self.threshold if element <= self.upper_integral_limit])
        if exists('k_distributions_table_3d_{}.json'.format(self.table_file_name)) and \
                exists('gamma_max_table_2d_{}.json'.format(self.table_file_name)) and \
                (not self.make_table_bool) is True:
            k_distributions_table_3d = json.load(open('k_distributions_table_3d_{}.json'.format(self.table_file_name)))
            gamma_max_table_2d = json.load(open('gamma_max_table_2d_{}.json'.format(self.table_file_name)))
        else:
            value = input('Table seems to be missing. Do you want to make one?(y/n)\n')
            if value == 'n':
                exit()
            self.make_table(self.make_table_bool, self.number_of_lists, self.table_file_name)
            # self.get_table(self.table_file_name+'.csv')  # TODO: not needed, since we set self.table in make_table now
            k_largest_intervals_table_3d = [[self._get_k_values(self.table[mu][n]) for n in range(len(self.table[mu]))] for mu in range(len(self.table))]
            k_distributions_table_3d = [self._get_k_distribution(k_largest_intervals_table_3d[mu]) for mu in range(len(k_largest_intervals_table_3d))]
            extremeness_table_3d = [[self._get_extremeness(k_largest_intervals_table_3d[mu][n], k_distributions_table_3d[mu]) for n in range(len(k_largest_intervals_table_3d[mu]))] for mu in range(len(k_largest_intervals_table_3d))]
            gamma_max_table_2d = [[max(extremeness_table_3d[mu][n]) for n in range(len(extremeness_table_3d[mu]))] for mu in range(len(extremeness_table_3d))]
            with open('k_distributions_table_3d_{}.json'.format(self.table_file_name), "w") as f:
                f.write(json.dumps(k_distributions_table_3d))
            with open('gamma_max_table_2d_{}.json'.format(self.table_file_name), "w") as f:
                f.write(json.dumps(gamma_max_table_2d))

        # TODO: delete code for plots
        # import matplotlib as mpl
        # mpl.rcParams['text.usetex'] = True
        # fig, ax = plt.subplots(1, 4, figsize=(16, 3.75), sharey=True)
        # for i, j in enumerate([500, 633, 766, 900]):
        #     ax[i].hist(gamma_max_table_2d[j], bins=25, density=True, cumulative=False)
        #     ax[i].set_xlim(0, 1)
        #     ax[i].set_xlabel(r'$\Gamma_{Max}$-values')
        #     ax[i].set_ylabel(r'Frequency')
        #     ax[i].set_title(r'$\Gamma_{Max}$-values '+f'for $\mu = {np.round(self.mus[j], 2)}$')
        #     ax[i].set_xticks(np.arange(0, 1.1, 0.1))
        #     ax[i].set_yticks([])
        # plt.tight_layout()
        # plt.savefig('hist_gamma.png', dpi=600)
        # plt.show()
        # # # # for i in range(900, 910):
        # # # #     plt.hist(gamma_max_table_2d[i], bins=75)
        # # # #     plt.show()
        # # fig, ax = plt.subplots(1, 4, figsize=(16, 3.75), sharey=True)
        # # for i, j in enumerate([0, 3, 6, 9]):
        # #     ax[i].hist(k_distributions_table_3d[900][j], bins=50, density=True, cumulative=False)
        # #     ax[i].set_xlim(0, 1)
        # #     ax[i].set_xlabel(r'$x$-values')
        # #     ax[i].set_ylabel(r'Frequency')
        # #     ax[i].set_title(f'$x$-values for $k = {j}$')
        # #     ax[i].set_xticks(np.arange(0, 1.1, 0.1))
        # #     ax[i].set_yticks([])
        # # plt.tight_layout()
        # # # plt.savefig('hist_x.png', dpi=600)
        # # plt.show()
        # #     # plt.figure(figsize=(4, 2.5*1.5))
        # #     # plt.hist(k_distributions_table_3d[900][i], bins=50, density=True, cumulative=False)
        # #     # plt.xlim(0, 1)
        # #     # # plt.ylim(0, 1)
        # #     # plt.xlabel(r'$x$-values')
        # #     # plt.ylabel(r'Frequency')
        # #     # plt.title(f'$x$-values for $k = {i}$')
        # #     # plt.xticks(np.arange(0, 1.1, 0.1))
        # #     # plt.yticks([])
        # #     # plt.tight_layout()
        # #     # # plt.savefig('cbarmax.png', dpi=300)
        # #     # plt.show()
        # exit()

        c_90percent_list = []
        c_90percent_list_2 = []
        sorted_list_gamma_max_table_mu = []
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

        masses_for_plot = []
        for mass in self.masses:
            print('Loop number =', list(self.masses).index(mass) + 1)
            print('mass =', mass)
            pdf = self.pdf(mass)
            pdf_sum = self.pdf_sum(pdf, self.materials)
            pdf_sum_convoluted = self.pdf_sum_convolution(pdf_sum)
            cdf_sum_convoluted = self.cdf_from_convoluted_pdf(pdf_sum_convoluted)
            self.cdf = np.copy(cdf_sum_convoluted)
            mus = self.get_mus(pdf_sum_convoluted, self.sigmas)
            self.mus_from_cdf = np.copy(mus)

            corresponding_cdf_values_data = self._get_corresponding_cdf_values(self.cdf)
            k_largest_intervals_data = self._get_k_values(corresponding_cdf_values_data)
            extremeness_data = [self._get_extremeness(k_largest_intervals_data, k_distributions_table_3d[mu]) for mu in range(len(k_distributions_table_3d))]
            cmaxs_data = [max(extremeness_data[mu]) for mu in range(len(extremeness_data))]
            cmaxs_data_extremeness = [self._get_extremeness(cmaxs_data[mu], gamma_max_table_2d[mu]) for mu in range(len(cmaxs_data))]
            y_filter = savgol_filter(cmaxs_data_extremeness, window_length=35, polyorder=3)

            # plt.plot(self.mus, cmaxs_data_extremeness, label='raw')
            # plt.plot(self.mus, y_filter, label='filter')
            # plt.legend()
            # plt.show()

            above_count, below_count = 0, 0
            there_are_values_above_90, there_are_values_below_90 = False, False
            for i in range(len(cmaxs_data_extremeness)):
                # if (not there_are_values_above_90 or not there_are_values_below_90) is True:
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
                new_x = sorted(list(cmaxs_data_extremeness)+list([self.cl]))
                y_interp = np.interp(new_x, cmaxs_data_extremeness, self.mus)
                mu_bar = y_interp[np.intersect1d(new_x, list([self.cl]), return_indices=True)[1]][0]
                sigma_bar = self._find_sigma_bar_from_mu_linear_model(mu_bar)
                print('mu_bar = ', mu_bar)
                print('sigma_bar = ', sigma_bar)
                # mu_index = 0
                # for i in range(len(cmaxs_data_extremeness)):
                #     if cmaxs_data_extremeness[i] >= self.cl:
                #         mu_index = i
                #         mu_bar = self.mus[mu_index]
                #         sigma_bar = self._find_sigma_bar_from_mu_linear_model(mu_bar)
                #         print('mu_index = ', mu_index)
                #         print('mu_bar = ', mu_bar)
                #         print('sigma_bar = ', sigma_bar)
                #         break
                new_x = sorted(list(y_filter)+list([self.cl]))
                y_interp = np.interp(new_x, y_filter, self.mus)
                mu_bar = y_interp[np.intersect1d(new_x, list([self.cl]), return_indices=True)[1]][0]
                sigma_bar = self._find_sigma_bar_from_mu_linear_model(mu_bar)
                print('mu_bar = ', mu_bar)
                print('sigma_bar = ', sigma_bar)
                masses_for_plot.append(mass)
                self.mus_corresponding_to_cbarmax_list.append(mu_bar)
                self.sigmas_corresponding_to_cbarmax_list.append(sigma_bar)
                # mu_index = 0
                # for i in range(len(y_filter)):
                #     if y_filter[i] >= self.cl:
                #         mu_index = i
                #         mu_bar = self.mus[mu_index]
                #         sigma_bar = self._find_sigma_bar_from_mu_linear_model(mu_bar)
                #         print('mu_index = ', mu_index)
                #         print('mu_bar (filter) = ', mu_bar)
                #         print('sigma_bar = ', sigma_bar)
                #         masses_for_plot.append(mass)
                #         self.mus_corresponding_to_cbarmax_list.append(mu_bar)
                #         self.sigmas_corresponding_to_cbarmax_list.append(sigma_bar)
                #         break
                #         # assert len(signal_pars) == len(self.table)
        # TODO: delete the part below
        plt.figure(figsize=[15., 10.])
        with open('C3P1_DetA_DataRelease_SI.xy', 'r', encoding='UTF8', newline='') as f:
            dataset = f.readlines()
            dataset = [line.strip('\n') for line in dataset if line[0] != '#']
        x_axis_plot = [float(number.split()[0]) for number in dataset]
        y_axis_plot = [float(number.split()[1]) for number in dataset]
        print('x_axis_plot =', x_axis_plot)
        print('y_axis_plot =', y_axis_plot)
        plt.plot(x_axis_plot, y_axis_plot, label='C3P1_DetA_DataRelease_SI.xy')
        plt.xscale('log')
        plt.yscale('log')
        # plt.legend()
        # plt.show()

        print('Acceptable masses =', masses_for_plot)
        print('Mus corresponding to cbarmax =', self.mus_corresponding_to_cbarmax_list)
        print('Sigmas corresponding to cbarmax =', self.sigmas_corresponding_to_cbarmax_list)
        plt.plot(masses_for_plot, self.sigmas_corresponding_to_cbarmax_list, label='exclusion chart sigma')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Mass (GeV)')
        plt.ylabel('Cross Section (pb)')
        plt.yticks(np.outer(np.logspace(-6, 3, 10), np.arange(1, 10, 1)).flatten())
        plt.grid(which='both', linestyle='dotted')
        plt.legend()
        plt.show()

        plt.plot(masses_for_plot, self.mus_corresponding_to_cbarmax_list, label='exclusion chart mu')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()

        pass

    def get_limit_from_other_nonlinear_model(self):
        """
        Calculate the limit for the cross-section of dark matter particles using Yellin's Optimum Interval Method.

        :return: None
        """
        self.data = np.array([element for element in self.data if element >= self.threshold if element <= self.upper_integral_limit])
        if exists('k_distributions_table_3d_{}.json'.format(self.table_file_name)) and \
                exists('gamma_max_table_2d_{}.json'.format(self.table_file_name)) and \
                (not self.make_table_bool) is True:
            k_distributions_table_3d = json.load(open('k_distributions_table_3d_{}.json'.format(self.table_file_name)))
            gamma_max_table_2d = json.load(open('gamma_max_table_2d_{}.json'.format(self.table_file_name)))
        else:
            self.make_table(self.make_table_bool, self.number_of_lists, self.table_file_name)
            # self.get_table(self.table_file_name+'.csv')
            k_largest_intervals_table_3d = [[self._get_k_values(self.table[mu][n]) for n in range(len(self.table[mu]))] for mu in range(len(self.table))]
            k_distributions_table_3d = [self._get_k_distribution(k_largest_intervals_table_3d[mu]) for mu in range(len(k_largest_intervals_table_3d))]
            extremeness_table_3d = [[self._get_extremeness(k_largest_intervals_table_3d[mu][n], k_distributions_table_3d[mu]) for n in range(len(k_largest_intervals_table_3d[mu]))] for mu in range(len(k_largest_intervals_table_3d))]
            gamma_max_table_2d = [[max(extremeness_table_3d[mu][n]) for n in range(len(extremeness_table_3d[mu]))] for mu in range(len(extremeness_table_3d))]
            with open('k_distributions_table_3d_{}.json'.format(self.table_file_name), "w") as f:
                f.write(json.dumps(k_distributions_table_3d))
            with open('gamma_max_table_2d_{}.json'.format(self.table_file_name), "w") as f:
                f.write(json.dumps(gamma_max_table_2d))

        c_90percent_list = []
        c_90percent_list_2 = []
        sorted_list_gamma_max_table_mu = []
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

        masses_for_plot = []
        for mass in self.masses:
            print('Loop number =', list(self.masses).index(mass) + 1)
            print('mass =', mass)
            cdf_for_this_mass = self.cdf[list(self.masses).index(mass)]

            corresponding_cdf_values_data = self._get_corresponding_cdf_values(cdf_for_this_mass)
            k_largest_intervals_data = self._get_k_values(corresponding_cdf_values_data)
            extremeness_data = [self._get_extremeness(k_largest_intervals_data, k_distributions_table_3d[mu]) for mu in range(len(k_distributions_table_3d))]
            cmaxs_data = [max(extremeness_data[mu]) for mu in range(len(extremeness_data))]
            cmaxs_data_extremeness = [self._get_extremeness(cmaxs_data[mu], gamma_max_table_2d[mu]) for mu in range(len(cmaxs_data))]
            y_filter = savgol_filter(cmaxs_data_extremeness, window_length=35, polyorder=3)

            plt.plot(self.mus, cmaxs_data_extremeness, label='raw')
            plt.plot(self.mus, y_filter, label='filter')
            plt.legend()
            plt.show()

            how_many_1, how_many_2 = 0, 0
            there_are_values_above_90, there_are_values_below_90 = False, False
            for i in range(len(cmaxs_data_extremeness)):
                if (not there_are_values_above_90 or not there_are_values_below_90) is True:
                    if cmaxs_data_extremeness[i] >= self.cl:
                        how_many_1 += 1
                        if how_many_1 == int(0.05*len(self.mus)):
                            there_are_values_above_90 = True
                    else:
                        how_many_2 += 1
                        if how_many_2 == int(0.05*len(self.mus)):
                            there_are_values_below_90 = True
            if there_are_values_above_90 == there_are_values_below_90 == True:
                print('Acceptable mu_bar can be found for this mass in this mu interval.')
            elif there_are_values_above_90 == True and there_are_values_below_90 == False:
                print('This mu interval is not suited for this mass, hence all cmax values seem to be above 90%.')
            elif there_are_values_above_90 == False and there_are_values_below_90 == True:
                print('This mu interval is not suited for this mass, hence all cmax values seem to be below 90%.')
            else:
                print('There seems to be a problem with the calculation of cmax values.')

            if there_are_values_above_90 == there_are_values_below_90 == True:
                new_x = sorted(list(cmaxs_data_extremeness)+list([self.cl]))
                y_interp = np.interp(new_x, cmaxs_data_extremeness, self.mus)
                mu_bar = y_interp[np.intersect1d(new_x, list([self.cl]), return_indices=True)[1]][0]
                sigma_bar = self._find_sigma_bar_from_mu_nonlinear_model(mu_bar)
                print('mu_bar = ', mu_bar)
                print('sigma_bar = ', sigma_bar)
                # mu_index = 0
                # for i in range(len(cmaxs_data_extremeness)):
                #     if cmaxs_data_extremeness[i] >= self.cl:
                #         mu_index = i
                #         # break
                #         mu_corresponding_to_cbarmax = self.mus[mu_index]
                #         sigma_corresponding_to_cbarmax = self._find_sigma_bar_from_mu_nonlinear_model(mu_corresponding_to_cbarmax)
                #         print('mu_index = ', mu_index)
                #         print('mu_corresponding_to_cbarmax = ', mu_corresponding_to_cbarmax)
                #         print('sigma_corresponding_to_cbarmax = ', sigma_corresponding_to_cbarmax)
                #         break
                new_x = sorted(list(y_filter)+list([self.cl]))
                y_interp = np.interp(new_x, y_filter, self.mus)
                mu_bar = y_interp[np.intersect1d(new_x, list([self.cl]), return_indices=True)[1]][0]
                sigma_bar = self._find_sigma_bar_from_mu_nonlinear_model(mu_bar)
                print('mu_bar = ', mu_bar)
                print('sigma_bar = ', sigma_bar)
                masses_for_plot.append(mass)
                self.mus_corresponding_to_cbarmax_list.append(mu_bar)
                self.sigmas_corresponding_to_cbarmax_list.append(sigma_bar)
            #     mu_index = 0
            #     for i in range(len(y_filter)):
            #         if y_filter[i] >= self.cl:
            #             mu_index = i
            #             # break
            #             mu_corresponding_to_cbarmax = self.mus[mu_index]
            #             sigma_corresponding_to_cbarmax = self._find_sigma_bar_from_mu_nonlinear_model(mu_corresponding_to_cbarmax)
            #             print('mu_index = ', mu_index)
            #             print('mu_corresponding_to_cbarmax (filter) = ', mu_corresponding_to_cbarmax)
            #             print('sigma_corresponding_to_cbarmax = ', sigma_corresponding_to_cbarmax)
            #             masses_for_plot.append(mass)
            #             self.mus_corresponding_to_cbarmax_list.append(mu_corresponding_to_cbarmax)
            #             self.sigmas_corresponding_to_cbarmax_list.append(sigma_corresponding_to_cbarmax)
            #             break
            #
            # # assert len(signal_pars) == len(self.table)

        print(self.masses)
        print(self.mus_corresponding_to_cbarmax_list)
        print(self.sigmas_corresponding_to_cbarmax_list)
        plt.plot(self.masses, self.sigmas_corresponding_to_cbarmax_list, label='exclusion chart sigma')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()

        plt.plot(self.masses, self.mus_corresponding_to_cbarmax_list, label='exclusion chart mu')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()

        pass

    def get_limit_from_other_linear_model(self):
        """
        Calculate the limit for the cross-section of dark matter particles using Yellin's Optimum Interval Method.

        :return: None
        """
        self.data = np.array([element for element in self.data if element >= self.threshold if element <= self.upper_integral_limit])
        if exists('k_distributions_table_3d_{}.json'.format(self.table_file_name)) and \
                exists('gamma_max_table_2d_{}.json'.format(self.table_file_name)) and \
                (not self.make_table_bool) is True:
            k_distributions_table_3d = json.load(open('k_distributions_table_3d_{}.json'.format(self.table_file_name)))
            gamma_max_table_2d = json.load(open('gamma_max_table_2d_{}.json'.format(self.table_file_name)))
        else:
            self.make_table(self.make_table_bool, self.number_of_lists, self.table_file_name)
            # self.get_table(self.table_file_name+'.csv')
            k_largest_intervals_table_3d = [[self._get_k_values(self.table[mu][n]) for n in range(len(self.table[mu]))] for mu in range(len(self.table))]
            k_distributions_table_3d = [self._get_k_distribution(k_largest_intervals_table_3d[mu]) for mu in range(len(k_largest_intervals_table_3d))]
            extremeness_table_3d = [[self._get_extremeness(k_largest_intervals_table_3d[mu][n], k_distributions_table_3d[mu]) for n in range(len(k_largest_intervals_table_3d[mu]))] for mu in range(len(k_largest_intervals_table_3d))]
            gamma_max_table_2d = [[max(extremeness_table_3d[mu][n]) for n in range(len(extremeness_table_3d[mu]))] for mu in range(len(extremeness_table_3d))]
            with open('k_distributions_table_3d_{}.json'.format(self.table_file_name), "w") as f:
                f.write(json.dumps(k_distributions_table_3d))
            with open('gamma_max_table_2d_{}.json'.format(self.table_file_name), "w") as f:
                f.write(json.dumps(gamma_max_table_2d))

        c_90percent_list = []
        c_90percent_list_2 = []
        sorted_list_gamma_max_table_mu = []
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

        masses_for_plot = []
        for mass in self.masses:  # TODO: sorted?
            print('Loop number =', list(self.masses).index(mass) + 1)
            print('mass =', mass)
            cdf_for_this_mass = self.cdf[list(self.masses).index(mass)]

            corresponding_cdf_values_data = self._get_corresponding_cdf_values(cdf_for_this_mass)
            k_largest_intervals_data = self._get_k_values(corresponding_cdf_values_data)
            extremeness_data = [self._get_extremeness(k_largest_intervals_data, k_distributions_table_3d[mu]) for mu in range(len(k_distributions_table_3d))]
            cmaxs_data = [max(extremeness_data[mu]) for mu in range(len(extremeness_data))]
            cmaxs_data_extremeness = [self._get_extremeness(cmaxs_data[mu], gamma_max_table_2d[mu]) for mu in range(len(cmaxs_data))]
            y_filter = savgol_filter(cmaxs_data_extremeness, window_length=35, polyorder=3)

            plt.plot(self.mus, cmaxs_data_extremeness, label='raw')
            plt.plot(self.mus, y_filter, label='filter')
            plt.legend()
            plt.show()

            how_many_1, how_many_2 = 0, 0
            there_are_values_above_90, there_are_values_below_90 = False, False
            for i in range(len(cmaxs_data_extremeness)):
                if (not there_are_values_above_90 or not there_are_values_below_90) is True:
                    if cmaxs_data_extremeness[i] >= self.cl:
                        how_many_1 += 1
                        if how_many_1 == int(0.05*len(self.mus)):
                            there_are_values_above_90 = True
                    else:
                        how_many_2 += 1
                        if how_many_2 == int(0.05*len(self.mus)):
                            there_are_values_below_90 = True
            if there_are_values_above_90 == there_are_values_below_90 == True:
                print('Acceptable mu_bar can be found for this mass in this mu interval.')
            elif there_are_values_above_90 == True and there_are_values_below_90 == False:
                print('This mu interval is not suited for this mass, hence all cmax values seem to be above 90%.')
            elif there_are_values_above_90 == False and there_are_values_below_90 == True:
                print('This mu interval is not suited for this mass, hence all cmax values seem to be below 90%.')
            else:
                print('There seems to be a problem with the calculation of cmax values.')

            if there_are_values_above_90 == there_are_values_below_90 == True:
                new_x = sorted(list(cmaxs_data_extremeness)+list([self.cl]))
                y_interp = np.interp(new_x, cmaxs_data_extremeness, self.mus)
                mu_bar = y_interp[np.intersect1d(new_x, list([self.cl]), return_indices=True)[1]][0]
                sigma_bar = self._find_sigma_bar_from_mu_linear_model(mu_bar)
                print('mu_bar = ', mu_bar)
                print('sigma_bar = ', sigma_bar)
                # mu_index = 0
                # for i in range(len(cmaxs_data_extremeness)):
                #     if cmaxs_data_extremeness[i] >= self.cl:
                #         mu_index = i
                #         # break
                #         mu_corresponding_to_cbarmax = self.mus[mu_index]
                #         sigma_corresponding_to_cbarmax = self._find_sigma_bar_from_mu_linear_model(mu_corresponding_to_cbarmax)
                #         print('mu_index = ', mu_index)
                #         print('mu_corresponding_to_cbarmax = ', mu_corresponding_to_cbarmax)
                #         print('sigma_corresponding_to_cbarmax = ', sigma_corresponding_to_cbarmax)
                #         break
                new_x = sorted(list(y_filter)+list([self.cl]))
                y_interp = np.interp(new_x, y_filter, self.mus)
                mu_bar = y_interp[np.intersect1d(new_x, list([self.cl]), return_indices=True)[1]][0]
                sigma_bar = self._find_sigma_bar_from_mu_linear_model(mu_bar)
                print('mu_bar = ', mu_bar)
                print('sigma_bar = ', sigma_bar)
                masses_for_plot.append(mass)
                self.mus_corresponding_to_cbarmax_list.append(mu_bar)
                self.sigmas_corresponding_to_cbarmax_list.append(sigma_bar)
            #     mu_index = 0
            #     for i in range(len(y_filter)):
            #         if y_filter[i] >= self.cl:
            #             mu_index = i
            #             # break
            #             mu_corresponding_to_cbarmax = self.mus[mu_index]
            #             sigma_corresponding_to_cbarmax = self._find_sigma_bar_from_mu_linear_model(mu_corresponding_to_cbarmax)
            #             print('mu_index = ', mu_index)
            #             print('mu_corresponding_to_cbarmax (filter) = ', mu_corresponding_to_cbarmax)
            #             print('sigma_corresponding_to_cbarmax = ', sigma_corresponding_to_cbarmax)
            #             masses_for_plot.append(mass)
            #             self.mus_corresponding_to_cbarmax_list.append(mu_corresponding_to_cbarmax)
            #             self.sigmas_corresponding_to_cbarmax_list.append(sigma_corresponding_to_cbarmax)
            #             break
            #
            # # assert len(signal_pars) == len(self.table)

        print(self.masses)
        print(self.mus_corresponding_to_cbarmax_list)
        print(self.sigmas_corresponding_to_cbarmax_list)
        plt.plot(self.masses, self.sigmas_corresponding_to_cbarmax_list, label='exclusion chart sigma')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()

        plt.plot(self.masses, self.mus_corresponding_to_cbarmax_list, label='exclusion chart mu')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()

        pass

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

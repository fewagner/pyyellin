import numpy as np
from scipy.signal import savgol_filter
from os.path import exists
import json
from matplotlib import pyplot as plt
import pyyellin as yell
# from SignalModel import SignalModel
# from Limit import Limit
import matplotlib as mpl


class Ilok(yell.SignalModel, yell.Limit):
    """
    The ultimate class that can calculate everything from SignalModel and from Limit.
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

    def set_sigmas(self, sigmas):
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

    def set_sigmas_max_gap(self, min_sigma, max_sigma, size):
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

    def set_mu_interval(self, min_mu, max_mu):
        """
        Set min and max values of the interval for expected number of events (μ).

        :param min_mu: Minimum expected number of events (μ). Minimum should be around Max(mus).
        :param max_mu: Maximum expected number of events (μ). Maximum should be around Max(mus).
        :return: None
        """
        self.mus = np.geomspace(min_mu, max_mu, 1000)  # TODO: change?
        # self.mus = np.linspace(min_mu, max_mu, 1000)
        return

    def get_limit_maximum_gap(self):  # TODO: this function and docstring of it
        self.data = np.array(self.data)
        self.data = self.data[(self.data > self.threshold) & (self.data < self.upper_integral_limit)]

        masses_for_plot = []
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
            ms = np.floor(mus/maximum_gap).astype(int)
            # c_zeros = [np.sum([((k * maximum_gap - mu)**k * np.exp(-k * maximum_gap))/long(np.math.factorial(k)) * (1 + k / (mu - k * maximum_gap)) for k in range(m+1)]) for m, mu in zip(ms, mus)]
            # print(mass, c_zeros, '\n\n')
            c_zeros = []
            for m, mu in zip(ms, mus):
                c_zero_per_k = []
                for k in range(int(m)+1):
                    try:
                        argument = ((k * maximum_gap - mu)**k * np.exp(-k * maximum_gap))/np.math.factorial(k) * (1 + k / (mu - k * maximum_gap))
                        print(k, argument)
                    except:
                        argument = 0
                    c_zero_per_k.append(argument)
                c_zeros.append(np.sum(c_zero_per_k))
            print(mass, c_zeros)

        return

    def get_limit(self):
        """
        Calculate the limit for the cross-section of dark matter particles using Yellin's Optimum Interval Method.

        :return: None
        """
        self.data = np.array(self.data)
        self.data = self.data[(self.data > self.threshold) & (self.data < self.upper_integral_limit)]

        if exists(f'x_distributions_table_3d_{self.table_file_name}.json') and \
                exists(f'gamma_max_table_2d_{self.table_file_name}.json') and \
                (not self.make_table_bool) is True:
            x_distributions_table_3d = json.load(open(f'x_distributions_table_3d_{self.table_file_name}.json'))
            gamma_max_table_2d = json.load(open(f'gamma_max_table_2d_{self.table_file_name}.json'))
        else:
            value = input('Table seems to be missing. Do you want to make one?(y/n)\n')
            if value == 'n':
                exit()
            self.make_table(self.make_table_bool, self.number_of_lists, self.table_file_name)
            x_values_table_3d = [[self._get_x_values(self.table[mu_index][n]) for n in range(len(self.table[mu_index]))] for mu_index in range(len(self.table))]
            x_distributions_table_3d = [self._get_x_distribution(x_values_table_3d[mu_index]) for mu_index in range(len(x_values_table_3d))]
            extremeness_table_3d = [[self._get_extremeness_of_x_values(x_values_table_3d[mu_index][n], x_distributions_table_3d[mu_index]) for n in range(len(x_values_table_3d[mu_index]))] for mu_index in range(len(x_values_table_3d))]
            # gamma_max_table_2d = [[max(extremeness_table_3d[mu_index][n]) for n in range(len(extremeness_table_3d[mu_index])) if len(extremeness_table_3d[mu_index][n])!=0] for mu_index in range(len(extremeness_table_3d))]  # TODO: remove if maybe, just trying a quick fix for the removal of added 0 and 1 to the random data, which makes is no longer Pois(mu) but Pois(mu+2) distributed
            gamma_max_table_2d = [[max(extremeness_table_3d[mu_index][n]) for n in range(len(extremeness_table_3d[mu_index]))] for mu_index in range(len(extremeness_table_3d))]
            with open(f'x_distributions_table_3d_{self.table_file_name}.json', "w") as f:
                f.write(json.dumps(x_distributions_table_3d))
            with open(f'gamma_max_table_2d_{self.table_file_name}.json', "w") as f:
                f.write(json.dumps(gamma_max_table_2d))


        # import matplotlib as mpl
        # import matplotlib.pylab as pylab
        # params = {'legend.fontsize': 'x-large',
        #           'figure.figsize': (15, 5),
        #           'axes.labelsize': 'x-large',
        #           'axes.titlesize':'x-large',
        #           'xtick.labelsize':'x-large',
        #           'ytick.labelsize':'x-large'}
        # pylab.rcParams.update(params)
        # mpl.rcParams['text.usetex'] = True
        # fig, axs = plt.subplots(2, 2, figsize=(12,9))
        # for i, j, k in zip([0,0,1,1], [0,1,0,1], [0,3,6,9]):
        #     axs[i,j].hist(x_distributions_table_3d[-300][k], bins=20)
        #     axs[i,j].set_xlim(0,1)
        #     axs[i,j].set_xlabel(r'$x$-values', fontsize=18)
        #     axs[i,j].set_ylabel(r'Count', fontsize=18)
        #     axs[i,j].set_title(r'$x$-values for k$=$'+f'{k}', fontsize=20)
        # plt.tight_layout(pad=1.2)
        # plt.savefig('hist_x_values.png', dpi=300)
        # plt.show()
        #
        # fig, axs = plt.subplots(2, 2, figsize=(12,9))
        # for i, j, k in zip([0,0,1,1], [0,1,0,1], [375, 550, 675, 775]):
        #     axs[i,j].hist(gamma_max_table_2d[k], bins=20)
        #     axs[i,j].set_xlim(0,1)
        #     axs[i,j].set_xlabel(r'$\Gamma_{Max}$-values', fontsize=18)
        #     axs[i,j].set_ylabel(r'Count', fontsize=18)
        #     axs[i,j].set_title(r'$\Gamma_{Max}$-values for $\mu=$'+f'{np.round(self.mus[k], 2)}',
        #                        fontsize=20)
        # plt.tight_layout(pad=1.2)
        # plt.savefig('hist_gamma_values.png', dpi=300)
        # plt.show()
        # exit()

        c_90percent_list = []
        # mu_90percent = []  # TODO: same reasoning as above, Pois(mu+2)
        for mu_index in range(len(gamma_max_table_2d)):
            # if len(gamma_max_table_2d[mu_index]) != 0:  # TODO: same reasoning as above, Pois(mu+2)
            c_90percent_list.append(np.percentile(gamma_max_table_2d[mu_index], self.cl*100))
            # mu_90percent.append(mu_index)  # TODO: same reasoning as above, Pois(mu+2)
        # for wl in [15, 20, 25, 30, 35]:
        # mpl.rcParams['text.usetex'] = True
        wl = 25
        c_90_savgol = savgol_filter(c_90percent_list, window_length=wl, polyorder=3)
        # plt.plot(self.mus, c_90percent_list, label=r'$\bar{C}_{Max}(0.9, \mu)$')
        plt.plot(self.mus, c_90percent_list, label='90% Cmax')
        plt.plot(self.mus, c_90_savgol, label='Savgol filter')
        plt.xscale('log')
        # plt.xlabel(r'Expected number of events $\mu$')
        # plt.ylabel(r'$\bar{C}_{Max}(0.9, \mu)$')
        plt.grid(which='both', linestyle='dotted')
        plt.legend()
        plt.savefig('c_90percent.png', dpi=300)
        plt.show()

        # mus_for_plot = []
        # mus_for_plot_2 = []
        # for mass in self.masses:
        #     pdf = self.pdf(mass)
        #     pdf_sum = self.pdf_sum(mass, pdf, self.materials)
        #     cdf_sum = self.cdf_from_pdf(pdf_sum)
        #     self.cdf = np.copy(cdf_sum)
        #     mus = self.get_mus(mass, pdf_sum, self.sigmas)
        #     mus_for_plot.append(mus[0])
        #     mus_for_plot_2.append(mus[1])
        # plt.plot(self.masses, mus_for_plot)
        # plt.show()
        # plt.plot(self.masses, mus_for_plot_2)
        # plt.show()

        masses_for_plot = []
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
            # cmaxs_data = [max(extremeness_data[mu]) if len(extremeness_data[mu])!=0 else 0 for mu in range(len(extremeness_data))]  # TODO: same reasoning as above, Pois(mu+2)
            cmaxs_data = [max(extremeness_data[mu]) for mu in range(len(extremeness_data))]
            cmaxs_data_extremeness = [self._get_extremeness_of_cmax(cmaxs_data[mu_index], gamma_max_table_2d[mu_index]) for mu_index in range(len(cmaxs_data))]

            # cmaxs_data_extremeness = np.array(cmaxs_data_extremeness)  # TODO: same reasoning as above, Pois(mu+2)
            # nan_indices = np.where(np.isnan(cmaxs_data_extremeness) == True)[0]
            # not_nan_indices = np.where(np.isnan(cmaxs_data_extremeness) == False)[0]
            # for i in range(len(nan_indices)):
            #     for j in range(len(not_nan_indices)):
            #         if nan_indices[i] > not_nan_indices[j]:
            #             k = not_nan_indices[j-1]
            #             l = not_nan_indices[j]
            #             cmaxs_data_extremeness[nan_indices[i]] = (cmaxs_data_extremeness[k] + cmaxs_data_extremeness[l])/2
            #             break  # TODO: same reasoning as above, Pois(mu+2)

            y_filter = savgol_filter(cmaxs_data_extremeness, window_length=35, polyorder=3)
            # plt.plot(self.mus, cmaxs_data_extremeness, label='raw')
            # plt.plot(self.mus, y_filter, label='filter')
            # plt.legend()
            # plt.show()

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
                new_x = sorted(list(cmaxs_data_extremeness)+[self.cl])
                y_interp = np.interp(new_x, cmaxs_data_extremeness, self.mus)
                mu_bar = y_interp[np.intersect1d(new_x, list([self.cl]), return_indices=True)[1]][0]
                sigma_bar = self._find_sigma_bar_from_mu_linear_model(mu_bar)
                print('mu_bar = ', mu_bar)
                print('sigma_bar = ', sigma_bar)

                new_x = sorted(list(y_filter)+[self.cl])
                y_interp = np.interp(new_x, y_filter, self.mus)
                mu_bar = y_interp[np.intersect1d(new_x, list([self.cl]), return_indices=True)[1]][0]
                sigma_bar = self._find_sigma_bar_from_mu_linear_model(mu_bar)
                print('mu_bar = ', mu_bar)
                print('sigma_bar = ', sigma_bar)

                y_filter_cmaxs_data_extremeness = savgol_filter(cmaxs_data_extremeness, window_length=35, polyorder=3)  # TODO: This sectio not compatible with Pois(mu) yet, instead with Pois(mu+2)
                y_filter_cmaxs_data = savgol_filter(cmaxs_data, window_length=35, polyorder=3)
                for mu_index in range(len(self.mus)):
                    if y_filter_cmaxs_data[mu_index] >= np.percentile(gamma_max_table_2d[mu_index], self.cl*100):
                        print(y_filter_cmaxs_data[mu_index], np.percentile(gamma_max_table_2d[mu_index], self.cl*100))
                        mu_bar = self.mus[mu_index]
                        break
                sigma_bar = self._find_sigma_bar_from_mu_linear_model(mu_bar)
                print('mu_bar = ', mu_bar)
                print('sigma_bar = ', sigma_bar)

                y_filter_cmaxs_data_extremeness = savgol_filter(cmaxs_data_extremeness, window_length=35, polyorder=3)
                y_filter_cmaxs_data = savgol_filter(cmaxs_data, window_length=35, polyorder=3)
                for mu_index in range(len(self.mus)):
                    if y_filter_cmaxs_data_extremeness[mu_index] >= np.percentile(gamma_max_table_2d[mu_index], self.cl*100):
                        print(y_filter_cmaxs_data_extremeness[mu_index], np.percentile(gamma_max_table_2d[mu_index], self.cl*100))
                        mu_bar = self.mus[mu_index]
                        break
                sigma_bar = self._find_sigma_bar_from_mu_linear_model(mu_bar)
                print('mu_bar = ', mu_bar)
                print('sigma_bar = ', sigma_bar)  # TODO: This sectio not compatible with Pois(mu) yet, instead with Pois(mu+2)

                masses_for_plot.append(mass)
                self.mus_corresponding_to_cbarmax_list.append(mu_bar)
                self.sigmas_corresponding_to_cbarmax_list.append(sigma_bar)
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

        # plt.figure(figsize=[15., 10.])
        # with open('lise/anc/Lise_TestCase.dat', 'r', encoding='UTF8', newline='') as f:
        #     dataset = f.readlines()
        #     dataset = [line.strip('\n') for line in dataset if line[0] != '#']
        # x_axis_plot = [float(number.split()[0]) for number in dataset]
        # y_axis_plot = np.array([float(number.split()[1]) for number in dataset])*1e36
        # print('x_axis_plot =', x_axis_plot)
        # print('y_axis_plot =', y_axis_plot)
        # plt.plot(x_axis_plot, y_axis_plot, label='Lise_TestCase.dat')
        # plt.xscale('log')
        # plt.yscale('log')
        # # plt.legend()
        # # plt.show()
        #
        # plt.figure(figsize=[15., 10.])
        # with open('lise/anc/CRESSTII-2015.dat.', 'r', encoding='UTF8', newline='') as f:
        #     dataset = f.readlines()
        #     dataset = [line.strip('\n') for line in dataset if line[0] != '#']
        # x_axis_plot = [float(number.split()[0]) for number in dataset]
        # y_axis_plot = np.array([float(number.split()[1]) for number in dataset])
        # print('x_axis_plot =', x_axis_plot)
        # print('y_axis_plot =', y_axis_plot)
        # plt.plot(x_axis_plot, y_axis_plot, label='Lise_Data_100_GeV.dat')
        # plt.xscale('log')
        # plt.yscale('log')

        print('Acceptable masses =', masses_for_plot)
        print('Mus corresponding to cbarmax =', self.mus_corresponding_to_cbarmax_list)
        print('Sigmas corresponding to cbarmax =', self.sigmas_corresponding_to_cbarmax_list)
        plt.plot(masses_for_plot, self.sigmas_corresponding_to_cbarmax_list, label='Cross sections determined by Ilok')
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
        plt.xlabel('Mass (GeV)')
        plt.ylabel('Expected number of events')
        plt.grid(which='both', linestyle='dotted')
        plt.legend()
        plt.show()

        return masses_for_plot, self.sigmas_corresponding_to_cbarmax_list, self.mus_corresponding_to_cbarmax_list

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
        :param mus: Expected number of events.
        :return: None
        """
        self.sigmas = sigmas
        self.mus_from_cdf = mus
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

    def get_limit_from_other_model(self, dependency_is_linear: bool):
        """
        Calculate the limit for the cross-section of dark matter particles using Yellin's Optimum Interval Method.

        :param dependency_is_linear: Whether the dependancy of mus on sigmas is linear or not.
        :return: None
        """
        self.data = np.array(self.data)
        self.data = self.data[(self.data > self.threshold) & (self.data < self.upper_integral_limit)]

        if exists(f'x_distributions_table_3d_{self.table_file_name}.json') and \
                exists(f'gamma_max_table_2d_{self.table_file_name}.json') and \
                (not self.make_table_bool) is True:
            x_distributions_table_3d = json.load(open(f'x_distributions_table_3d_{self.table_file_name}.json'))
            gamma_max_table_2d = json.load(open(f'gamma_max_table_2d_{self.table_file_name}.json'))
        else:
            value = input('Table seems to be missing. Do you want to make one?(y/n)\n')
            if value == 'n':
                exit()
            self.make_table(self.make_table_bool, self.number_of_lists, self.table_file_name)
            x_values_table_3d = [[self._get_x_values(self.table[mu_index][n]) for n in range(len(self.table[mu_index]))] for mu_index in range(len(self.table))]
            x_distributions_table_3d = [self._get_x_distribution(x_values_table_3d[mu_index]) for mu_index in range(len(x_values_table_3d))]
            extremeness_table_3d = [[self._get_extremeness_of_x_values(x_values_table_3d[mu_index][n], x_distributions_table_3d[mu_index]) for n in range(len(x_values_table_3d[mu_index]))] for mu_index in range(len(x_values_table_3d))]
            # gamma_max_table_2d = [[max(extremeness_table_3d[mu_index][n]) for n in range(len(extremeness_table_3d[mu_index])) if len(extremeness_table_3d[mu_index][n])!=0] for mu_index in range(len(extremeness_table_3d))]  # TODO: remove if maybe, just trying a quick fix for the removal of added 0 and 1 to the random data, which makes is no longer Pois(mu) but Pois(mu+2) distributed
            gamma_max_table_2d = [[max(extremeness_table_3d[mu_index][n]) for n in range(len(extremeness_table_3d[mu_index]))] for mu_index in range(len(extremeness_table_3d))]
            with open(f'x_distributions_table_3d_{self.table_file_name}.json', "w") as f:
                f.write(json.dumps(x_distributions_table_3d))
            with open(f'gamma_max_table_2d_{self.table_file_name}.json', "w") as f:
                f.write(json.dumps(gamma_max_table_2d))

        c_90percent_list = []
        # mu_90percent = []  # TODO: same reasoning as above, Pois(mu+2)
        for mu_index in range(len(gamma_max_table_2d)):
            # if len(gamma_max_table_2d[mu_index]) != 0:  # TODO: same reasoning as above, Pois(mu+2)
            c_90percent_list.append(np.percentile(gamma_max_table_2d[mu_index], self.cl*100))
            # mu_90percent.append(mu_index)  # TODO: same reasoning as above, Pois(mu+2)
        # for wl in [15, 20, 25, 30, 35]:
        wl = 25
        c_90_savgol = savgol_filter(c_90percent_list, window_length=wl, polyorder=3)
        plt.plot(self.mus, c_90percent_list, label='90% Cmax List')
        plt.plot(self.mus, c_90_savgol, label=f'savgol {wl}')
        plt.xscale('log')
        plt.legend()
        plt.show()

        # mus_for_plot = []
        # mus_for_plot_2 = []
        # for mass in self.masses:
        #     pdf = self.pdf(mass)
        #     pdf_sum = self.pdf_sum(mass, pdf, self.materials)
        #     cdf_sum = self.cdf_from_pdf(pdf_sum)
        #     self.cdf = np.copy(cdf_sum)
        #     mus = self.get_mus(mass, pdf_sum, self.sigmas)
        #     mus_for_plot.append(mus[0])
        #     mus_for_plot_2.append(mus[1])
        # plt.plot(self.masses, mus_for_plot)
        # plt.show()
        # plt.plot(self.masses, mus_for_plot_2)
        # plt.show()

        masses_for_plot = []
        for mass in self.masses:
            print('Loop number =', list(self.masses).index(mass) + 1)
            print('mass =', mass)
            cdf_for_this_mass = self.cdf[list(self.masses).index(mass)]

            corresponding_cdf_values_data = self._get_corresponding_cdf_values(self.cdf)
            x_values_data = self._get_x_values(corresponding_cdf_values_data)
            extremeness_data = [self._get_extremeness_of_x_values(x_values_data, x_distributions_table_3d[mu_index]) for mu_index in range(len(x_distributions_table_3d))]
            # cmaxs_data = [max(extremeness_data[mu]) if len(extremeness_data[mu])!=0 else 0 for mu in range(len(extremeness_data))]  # TODO: same reasoning as above, Pois(mu+2)
            cmaxs_data = [max(extremeness_data[mu]) for mu in range(len(extremeness_data))]
            cmaxs_data_extremeness = [self._get_extremeness_of_cmax(cmaxs_data[mu_index], gamma_max_table_2d[mu_index]) for mu_index in range(len(cmaxs_data))]

            # cmaxs_data_extremeness = np.array(cmaxs_data_extremeness)  # TODO: same reasoning as above, Pois(mu+2)
            # nan_indices = np.where(np.isnan(cmaxs_data_extremeness) == True)[0]
            # not_nan_indices = np.where(np.isnan(cmaxs_data_extremeness) == False)[0]
            # for i in range(len(nan_indices)):
            #     for j in range(len(not_nan_indices)):
            #         if nan_indices[i] > not_nan_indices[j]:
            #             k = not_nan_indices[j-1]
            #             l = not_nan_indices[j]
            #             cmaxs_data_extremeness[nan_indices[i]] = (cmaxs_data_extremeness[k] + cmaxs_data_extremeness[l])/2
            #             break  # TODO: same reasoning as above, Pois(mu+2)

            y_filter = savgol_filter(cmaxs_data_extremeness, window_length=35, polyorder=3)
            # plt.plot(self.mus, cmaxs_data_extremeness, label='raw')
            # plt.plot(self.mus, y_filter, label='filter')
            # plt.legend()
            # plt.show()

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
                new_x = sorted(list(cmaxs_data_extremeness)+list([self.cl]))
                y_interp = np.interp(new_x, cmaxs_data_extremeness, self.mus)
                mu_bar = y_interp[np.intersect1d(new_x, list([self.cl]), return_indices=True)[1]][0]
                if dependency_is_linear is True:
                    sigma_bar = self._find_sigma_bar_from_mu_linear_model(mu_bar)
                else:
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
                if dependency_is_linear is True:
                    sigma_bar = self._find_sigma_bar_from_mu_linear_model(mu_bar)
                else:
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
        plt.plot(masses_for_plot, self.sigmas_corresponding_to_cbarmax_list, label='exclusion chart sigma')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()

        plt.plot(masses_for_plot, self.mus_corresponding_to_cbarmax_list, label='exclusion chart mu')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()
        plt.show()

        return masses_for_plot, self.sigmas_corresponding_to_cbarmax_list, self.mus_corresponding_to_cbarmax_list

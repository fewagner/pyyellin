import random
import numpy as np
from scipy import integrate
from scipy.stats import norm
from scipy.special import erf
from scipy import constants as const


class SignalModel:
    """
    A class that models the expected signal based on WIMP Model. The WIMP-nucleus cross-section can
    be separated into a spin-independent and a spin-dependent contribution. If the target
    nuclei have no or only a small net spin as it is the case in the CRESST experiment (with CaWO4 as a target
    material), the spin-dependent part can be neglected. And thus, the signal model calculated in the following
    class focuses on the spin-independent part of the WIMP-nucleus cross-section. Pdf, cdf and rvs can be calculated
    with the help of setup functions set_detector and set_cut_eff.
    """
    R_Ca_O = 8e-15/(const.hbar*const.c/const.e)*1e3
    av_dict = {'Ca': {1: 0.44846e-1, 2: 0.61326e-1, 3: -0.16818e-2, 4: -0.26217e-1, 5: -0.29725e-2, 6: 0.85534e-2,
                      7: 0.35322e-2, 8: -0.48258e-3, 9: -0.39346e-3, 10: 0.20338e-3, 11: 0.25461e-4,
                      12: -0.17794e-4, 13: 0.67394e-5, 14: -0.21033e-5},
               'O': {1: 0.20238e-1, 2: 0.44793e-1, 3: 0.33533e-1, 4: 0.35030e-2, 5: -0.12293e-1, 6: -0.10329e-1,
                     7: -0.34036e-2, 8: -0.41627e-3, 9: -0.94435e-3, 10: -0.25771e-3, 11: 0.23759e-3,
                     12: -0.10603e-3, 13: 0.41480e-4}}

    def __init__(self):
        """
        Here global parameters for the signal model can be fixed. E.g. the escape velocity, ...
        """
        self.METERS_TO_EV = const.hbar*const.c/const.e  # conversion factor to convert meters in c*hbar/eV
        self.DIMENSION_FACTOR = (const.e*10**19)**2*86400*10**30/(const.hbar*10**34)**3
        self.RHO_X = 0.3e3*self.METERS_TO_EV**3  # density [keV^4/(c^5*hbar^3)]
        self.M_P = const.physical_constants["proton mass energy equivalent in MeV"][0]*10**3  # proton mass [keV/c^2]
        self.S = 1e-15/self.METERS_TO_EV*1e3  # skin thickness [c*hbar/keV]
        self.V_ESC = 550000./const.c  # escape velocity [c]
        self.W = 270000./const.c  # root mean square velocity of the DM particles [c]
        self.V_EARTH = 220*1.05e3/const.c  # earth velocity [c]
        self.F_A = .52e-15/self.METERS_TO_EV*1e3  # factor for nuclear radius r_0 [c*hbar/keV]
        self.F_S = .9e-15/self.METERS_TO_EV*1e3  # factor for nuclear radius r_0 [c*hbar/keV]
        self.exposure = 0.
        self.resolution = 0.
        self.threshold = 0.
        self.upper_integral_limit = 0.
        self.energy_grid_step_size = 0.0005
        self.energy_grid_upper_bound = 0.
        self.materials = []
        self.pdf_sum_normalization_factor = {}
        self.normalization_list = {}
        self.cut_eff = []
        self.material_AR_eff = []
        self.material_cut_eff = []
        self.energy_grid = []

    def set_detector(self, exposure: float, resolution: float, threshold: float, materials: list,
                     upper_integral_limit: float, energy_grid_step_size: float = 0.0005,
                     energy_grid_upper_bound: float = 1000.):
        """
        Set various parameters of the detector.

        :param exposure: Exposure [kg*d]
        :param resolution: Resolution [keV]
        :param threshold: Threshold [keV]
        :param materials: List of materials with the information: mass, percentile mass in the molecule, element name
        :param upper_integral_limit: Upper integral limit, energy range restriction [keV]
        :param energy_grid_step_size: Step size for the energy grid [keV]
        :param energy_grid_upper_bound: Upper bound for energy grid. [keV]
        :return: None
        """

        self.exposure = exposure
        self.resolution = resolution
        self.threshold = threshold
        self.materials = materials
        self.upper_integral_limit = upper_integral_limit
        self.energy_grid_step_size = energy_grid_step_size
        self.energy_grid = np.arange(threshold-3*resolution, 200, energy_grid_step_size)
        self.energy_grid_upper_bound = energy_grid_upper_bound
        self._set_material_AR_eff()
        return

    def _set_material_AR_eff(self):
        """
        Set probability of being in the acceptance region for each material.

        :return: None
        """
        for i in range(len(self.materials)):
            x_axis = []
            y_axis = []
            with open(self.materials[i][-1], 'r', encoding='UTF8', newline='') as f:
                dataset = f.readlines()
                dataset = [line.strip('\n') for line in dataset if line[0] != '#']
                dataset = [line.split('\t') for line in dataset if line[0] != '#']
            x_axis.append([float(number[0]) for number in dataset])
            y_axis.append([float(number[1]) for number in dataset])
            energy_grid = np.arange(self.threshold-3*self.resolution, self.energy_grid_upper_bound, self.energy_grid_step_size)
            y_interp = np.interp(energy_grid, x_axis[0], y_axis[0])
            self.material_AR_eff.append(y_interp)
        return

    def set_cut_eff(self, file_name):
        """
        Set cut efficiency.

        :param file_name: File path and name. If set to 1, then cut efficiency will be set to a list consisting of ones.
        :return: None
        """
        # if type(file_name) == str:
        #     with open(file_name, 'r', encoding='UTF8', newline='') as f:
        #         dataset = f.readlines()
        #         dataset = [line.strip('\n') for line in dataset if line[0] != '#']
        #         dataset = [line.split('\t') for line in dataset if line[0] != '#']
        #     self.cut_eff.append([float(number[0]) for number in dataset])
        #     self.cut_eff.append([float(number[1]) for number in dataset])
        #     energy_grid = np.arange(self.threshold-3*self.resolution, self.energy_grid_upper_bound, self.energy_grid_step_size)
        #     y_interp = np.interp(energy_grid, self.cut_eff[0], self.cut_eff[1])
        #     self.cut_eff = np.copy(y_interp)
        # elif file_name == 1:
        #     self.cut_eff = [1 for i in range(len(np.arange(self.threshold-3*self.resolution, self.energy_grid_upper_bound, self.energy_grid_step_size)))]
        # else:
        #     print('Parameter should either be the file name in string format or the integer 1.')
        if file_name == 1:
            self.cut_eff = [1 for i in range(len(np.arange(self.threshold-3*self.resolution, self.energy_grid_upper_bound, self.energy_grid_step_size)))]
        else:
            with open(file_name, 'r', encoding='UTF8', newline='') as f:
                dataset = f.readlines()
                dataset = [line.strip('\n') for line in dataset if line[0] != '#']
                dataset = [line.split('\t') for line in dataset if line[0] != '#']
            self.cut_eff.append([float(number[0]) for number in dataset])
            self.cut_eff.append([float(number[1]) for number in dataset])
            energy_grid = np.arange(self.threshold-3*self.resolution, self.energy_grid_upper_bound, self.energy_grid_step_size)
            y_interp = np.interp(energy_grid, self.cut_eff[0], self.cut_eff[1])
            self.cut_eff = np.copy(y_interp)
        return

    def convolve_pdf(self, pdf):
        """
        Calculate the convolution of the pdf with a gaussian function in order to take into account the finite energy resolution.
        :param pdf: The probability density function.
        :return: Convolved pdf.
        """
        gauss_x = np.arange(-10*self.resolution, 10*self.resolution, 0.0005)
        if len(gauss_x) >= len(pdf):
            gauss_x = np.linspace(-10*self.resolution, 10*self.resolution, len(pdf)-2)
        gauss = norm.pdf(gauss_x, scale=self.resolution)
        convolved_pdf = np.convolve(pdf, gauss, mode='same')/np.sum(gauss)
        return convolved_pdf

    def pdf(self, pars: float):
        """
        Calculate the probability density function of the signal model, evaluated at the set energy grid.

        :param pars: The mass of the dark matter particle.
        :return: The energy grid and the probability density functions evaluated on the grid.
        :rtype: list
        """
        rates_per_energy_list = []
        self.normalization_list[pars] = {}

        energies_x_min_3 = []
        for element in self.materials:
            energy_grid_mock = np.arange(self.threshold-3*self.resolution, self.energy_grid_upper_bound, 0.05)
            energy_grid_mock = energy_grid_mock[energy_grid_mock > 0]
            A = element[0]
            M_N = A*self.M_P
            m_chi = np.copy(pars)*1e6
            mu_N = M_N*m_chi/(M_N+m_chi)
            v_min = np.sqrt(energy_grid_mock * M_N / (2 * mu_N ** 2))
            x_min = np.sqrt(3*v_min**2/(2*self.W**2))
            z = np.sqrt(3/2)*self.V_ESC/self.W
            eta = np.sqrt(3/2)*self.V_EARTH/self.W
            x_min_3_mock_list = [item for item in x_min if item >= (z+eta)]
            energies_x_min_3.append(energy_grid_mock[len(energy_grid_mock)-len(x_min_3_mock_list)])
        print(f'm_chi: {np.copy(pars)} GeV')
        print('max(energies_x_min_3):', max(energies_x_min_3))
        if max(energies_x_min_3) < 1:
            max_energy = 1.
        elif max(energies_x_min_3) >= self.energy_grid_upper_bound:
            max_energy = self.energy_grid_upper_bound
        else:
            max_energy = max(energies_x_min_3)
        self.energy_grid = np.arange(self.threshold-3*self.resolution, max_energy, self.energy_grid_step_size)
        self.energy_grid = self.energy_grid[self.energy_grid > 0]

        for element in self.materials:
            A = element[0]
            M_N = A * self.M_P
            F_C = (1.23*A**(1./3.)-.6)*1e-15/self.METERS_TO_EV*1e3
            m_chi = np.copy(pars)*1e6
            mu_p = self.M_P*m_chi/(self.M_P+m_chi)
            mu_N = M_N*m_chi/(M_N+m_chi)
            z = np.sqrt(3/2)*self.V_ESC/self.W
            n = erf(z)-2/np.sqrt(np.pi)*z*np.exp(-z**2)
            r_0 = np.sqrt((F_C**2)+7./3.*(np.pi**2)*(self.F_A**2)-5.*(self.F_S**2))
            eta = np.sqrt(3/2)*self.V_EARTH/self.W
            q = np.sqrt(2 * M_N * self.energy_grid)
            spherical_bessel_j1 = (np.sin(q*r_0)/(q*r_0)**2)-(np.cos(q*r_0)/(q*r_0))
            f = 3.*spherical_bessel_j1/(q*r_0)*np.exp(-0.5*q*q*self.S*self.S)
            v_min = np.sqrt(self.energy_grid*M_N/(2*mu_N**2))
            x_min = np.array(np.sqrt(3/2)*v_min/self.W)
            x_min_1 = x_min[x_min < (z-eta)]
            x_min_2 = x_min[(x_min >= (z-eta)) & (x_min < (z+eta))]
            x_min_3 = x_min[x_min >= (z+eta)]
            integral_1 = np.sqrt(np.pi)/2*(erf(x_min_1+eta)-erf(x_min_1-eta))-2*eta*np.exp(-z**2)
            integral_2 = np.sqrt(np.pi)/2*(erf(z)-erf(x_min_2-eta))-np.exp(-z**2)*(z+eta-x_min_2)
            integral_3 = np.zeros(len(x_min_3))
            integral = np.concatenate((integral_1, integral_2, integral_3), axis=None)
            integral = integral * 1/(n*eta)*np.sqrt(3/(2*np.pi*self.W**2))
            rates_per_energy = self.RHO_X/(2.*mu_p**2*m_chi)*A**2*f**2*integral
            rates_per_energy = rates_per_energy * self.DIMENSION_FACTOR
            rates_per_energy = self.convolve_pdf(rates_per_energy)
            rates_per_energy = rates_per_energy*self.material_AR_eff[list(self.materials).index(element)][:len(rates_per_energy)]
            normalization = integrate.trapz(rates_per_energy, self.energy_grid)
            if normalization != 0:
                rates_per_energy = rates_per_energy/normalization
            rates_per_energy_list.append(rates_per_energy)
            self.normalization_list[pars][element[2]] = normalization
        return self.energy_grid, rates_per_energy_list

    def pdf_sum_from_pdf(self, mass: float, pdf: list, materials: list):
        """
        Sum the pdfs of the given materials according to their percentage in the molecule.

        :param mass: The mass of the dark matter particle.
        :param pdf: The energy grid and the probability density functions.
        :param materials: List of materials with the information: mass, percentile mass in the molecule, element name
        :return: Energy grid and summed probability density function.
        :rtype: list
        """
        pdf_sum = np.zeros(np.shape(pdf[0])[0])
        for i in range(len(pdf[1])):
            pdf_sum = np.nansum([pdf_sum, pdf[1][i]*materials[i][1]*self.normalization_list[mass][materials[i][2]]], axis=0)
        self.pdf_sum_normalization_factor[mass] = integrate.trapz(pdf_sum, self.energy_grid)
        pdf_sum = pdf_sum/self.pdf_sum_normalization_factor[mass]
        return pdf[0], pdf_sum

    @staticmethod
    def cdf_sum_from_pdf_sum(pdf_sum):
        """
        Sum the cdfs of the given materials according to their percentage in the molecule.

        :param pdf_sum: Energy grid and summed probability density function.
        :return: Energy grid and summed cumulative density function.
        :rtype: list
        """
        cdf = np.zeros(np.shape(pdf_sum[0])[0])
        for i in range(np.shape(pdf_sum[0])[0]-1):
            cdf[i+1] = cdf[i] + integrate.trapz(pdf_sum[1][i:i+2], pdf_sum[0][i:i+2])
        return pdf_sum[0], cdf

    def get_mus(self, mass: float, pdf_sum: list, sigmas: list):
        """
        Calculate the expected amount of events μ.

        :param mass: The mass of the dark matter particle.
        :param pdf_sum: The summed probability density function.
        :param sigmas: Cross-sections [pbarn].
        :return: Expected amount of events μ.
        :rtype: numpy array
        """
        pdf = pdf_sum[1]*self.cut_eff[:len(pdf_sum[1])]
        pdf_sum_limited_x = pdf_sum[0][np.logical_and(pdf_sum[0] > self.threshold, pdf_sum[0] < self.upper_integral_limit)]
        pdf_sum_limited_y = pdf[np.logical_and(pdf_sum[0] > self.threshold, pdf_sum[0] < self.upper_integral_limit)]
        mus = integrate.trapz(pdf_sum_limited_y, pdf_sum_limited_x)*np.array(sigmas)*self.pdf_sum_normalization_factor[mass]*self.exposure
        return mus

    ### Additional Functions ###

    def pdf_sum(self, mass: float, materials: list):
        """
        Sum the pdfs of the given materials according to their percentage in the molecule.

        :param mass: The mass of the dark matter particle.
        :param pdf: The energy grid and the probability density functions.
        :param materials: List of materials with the information: mass, percentile mass in the molecule, element name
        :return: Energy grid and summed probability density function.
        :rtype: list
        """
        pdf = self.pdf(mass*1e6)
        pdf_sum = np.zeros(np.shape(pdf[0])[0])
        for i in range(len(pdf[1])):
            pdf_sum = np.nansum([pdf_sum, pdf[1][i]*materials[i][1]*self.normalization_list[mass][materials[i][2]]], axis=0)
        self.pdf_sum_normalization_factor[mass] = integrate.trapz(pdf_sum, self.energy_grid)
        pdf_sum = pdf_sum/self.pdf_sum_normalization_factor[mass]
        return pdf[0], pdf_sum

    def cdf(self, pars: float):
        """
        Calculate the cumulative density function of the signal model, evaluated at the set energy grid.

        :param pars: The mass of the dark matter particle.
        :return: The energy grid and the cumulative density functions evaluated on the grid.
        :rtype: list
        """
        m_chi = np.copy(pars)*1e6
        pdf = self.pdf(m_chi)
        cdf = np.zeros((np.shape(self.materials)[0], np.shape(pdf[0])[0]))
        for j in range(len(self.materials)):
            for i in range(np.shape(pdf[0])[0]-1):
                cdf[j][i+1] = cdf[j][i]+integrate.trapz(pdf[1][j][i:i+2], pdf[0][i:i+2])
        return pdf[0], cdf

    def cdf_from_pdf(self, pdf: list):
        """
        Calculate the cumulative density function of the signal model, evaluated at the set energy grid.

        :param pdf: The energy grid and the probability density function.
        :return: The energy grid and the cumulative density functions evaluated on the grid.
        :rtype: list
        """
        cdf = np.zeros((np.shape(self.materials)[0], np.shape(pdf[0])[0]))
        for j in range(len(self.materials)):
            for i in range(np.shape(pdf[0])[0]-1):
                cdf[j][i+1] = cdf[j][i]+integrate.trapz(pdf[1][j][i:i+2], pdf[0][i:i+2])
        return pdf[0], cdf

    def cdf_sum(self, pars: float):
        """
        Sum the cdfs of the given materials according to their percentage in the molecule.

        :param cdf: The energy grid and the probability density functions.
        :param materials: List of materials with the information: mass, percentile mass in the molecule, element name
        :return: Energy grid and summed cumulative density function.
        :rtype: list
        """
        m_chi = np.copy(pars)*1e6
        pdf = self.pdf(m_chi)
        pdf_sum = self.pdf_sum_from_pdf(m_chi, pdf, self.materials)
        cdf_sum = self.cdf_sum_from_pdf_sum(pdf_sum)
        return cdf_sum

    def rvs(self, size: int, pars: float):
        """
        Draw a sample from the signal model. Inputs are the size and the mass of the dark matter.

        :param size: The sample size
        :param pars: The mass of the dark matter particle.
        :return: The recoil energies of the drawn sample.
        :rtype: numpy array
        """
        cdf = self.cdf(pars*1e6)
        sample = []
        samples = []
        for k in range(len(self.materials)):
            for i in range(size):
                rand = random.random()
                for j in range(len(cdf[0])):
                    if cdf[1][k][j] >= rand:
                        sample.append(cdf[0][j-1])
                        break
            samples.append(sample)
        return np.array(samples)

    def rvs_from_cdf(self, size: int, cdf: list):
        """
        Draw a sample from the signal model. Inputs are the size and the cdf.

        :param size: The sample size
        :param cdf: Energy grid and the cumulative density function evaluated at the energy grid.
        :return: The recoil energies of the drawn sample.
        :rtype: numpy array
        """
        sample = []
        samples = []
        for k in range(len(self.materials)):
            for i in range(size):
                rand = random.random()
                for j in range(len(cdf[0])):
                    if cdf[1][k][j] >= rand:
                        sample.append(cdf[0][j-1])
                        break
            samples.append(sample)
        return np.array(samples)

    def rvs_sum(self, size: int, pars: float):
        """
        Draw a sample from the signal model using the summed cdf.

        :param size: The sample size
        :param cdf_sum: Energy grid and the summed cumulative density function evaluated at the energy grid.
        :return: The recoil energies of the drawn sample.
        :rtype: numpy array
        """
        cdf_sum = self.cdf_sum(pars*1e6)
        sample = []
        for i in range(size):
            rand = random.random()
            for j in range(len(cdf_sum[0])):
                if cdf_sum[1][j] >= rand:
                    sample.append(cdf_sum[0][j-1])
                    break
        return np.array(sample)

    @staticmethod
    def rvs_sum_from_cdf_sum(size: int, cdf_sum: list):
        """
        Draw a sample from the signal model using the summed cdf.

        :param size: The sample size
        :param cdf_sum: Energy grid and the summed cumulative density function evaluated at the energy grid.
        :return: The recoil energies of the drawn sample.
        :rtype: numpy array
        """
        sample = []
        for i in range(size):
            rand = random.random()
            for j in range(len(cdf_sum[0])):
                if cdf_sum[1][j] >= rand:
                    sample.append(cdf_sum[0][j-1])
                    break
        return np.array(sample)

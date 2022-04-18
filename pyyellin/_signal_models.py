import random
import numpy as np
from scipy import integrate
from scipy import constants as const
from scipy.special import erf
from scipy.stats import norm
from scipy import optimize


class SignalModel:
    """
    A class that models the expected signal based on WIMP Model. The WIMP-nucleus cross-section can
    be separated into a spin-independent and a spin-dependent contribution. If the target
    nuclei have no or only a small net spin as it is the case in the CRESST experiment (with CaWO4 as a target
    material), the spin-dependent part can be neglected. And thus, the signal model calculated in the following
    class focuses on the spin-independent part of the WIMP-nucleus cross-section. Pdf, cdf and rvs can be calculated
    with the help of setup functions set_detector and set_cut_eff.
    """

    def __init__(self):
        """
        Here global parameters for the signal model can be fixed. E.g. the escape velocity, ...
        """
        self.METERS_TO_EV = const.hbar*const.c/const.e  # conversion factor to convert meters in c*hbar/eV
        self.DIMENSION_FACTOR = (const.e*10**19)**2*86400*10**30/(const.hbar*10**34)**3  # TODO: dimension ???
        self.RHO_X = 0.3e3*self.METERS_TO_EV**3  # density [keV^4/(c^5*hbar^3)]
        self.M_P = const.physical_constants["proton mass energy equivalent in MeV"][0]*10**3  # proton mass [keV/c^2]  # TODO: amu in GeV/c^2 oder mp passt
        self.S = 1e-15/self.METERS_TO_EV*1e3  # skin thickness [c*hbar/keV]
        self.V_ESC = 554000./const.c  # escape velocity [c]
        self.W = 270000./const.c  # root mean square velocity of the DM particles [c]
        self.V_EARTH = 220*1.05e3/const.c  # earth velocity [c]
        self.F_A = .52e-15/self.METERS_TO_EV*1e3  # factor for nuclear radius r_0 [c*hbar/keV]
        self.F_S = .9e-15/self.METERS_TO_EV*1e3  # factor for nuclear radius r_0 [c*hbar/keV]
        self.exposure = 0.
        self.resolution = 0.
        self.threshold = 0.
        self.upper_integral_limit = 0.
        self.materials = []
        self.normalization_factor = 1.
        self.normalization_list = []
        self.cut_eff = []
        self.energy_grid = []

    def set_detector(self, exposure: float, resolution: float, threshold: float, materials: list,
                     upper_integral_limit: float, energy_grid_step_size: float = 0.0005):
        """
        Set various parameters of the detector.

        :param exposure: Exposure [kg*d]
        :param resolution: Resolution [keV]
        :param threshold: Threshold [keV]
        :param materials: List of materials with the information: mass, percentile mass in the molecule, element name
        :param upper_integral_limit: Upper integral limit, energy range restriction [keV]
        :param energy_grid_step_size: Step size for the energy grid [keV]
        :return: None
        """

        self.exposure = exposure
        self.resolution = resolution  # scale (sigma) von gauss
        self.threshold = threshold
        self.materials = materials
        self.upper_integral_limit = upper_integral_limit
        self.energy_grid = np.arange(threshold-3*resolution, upper_integral_limit+3*resolution, energy_grid_step_size)
        return

    def pdf(self, pars: float):
        """
        Calculate the probability density function of the signal model, evaluated at the set energy grid.

        :param pars: The mass of the dark matter particle.
        :return: The energy grid and the probability density functions evaluated on the grid.
        :rtype: list
        """
        rates_per_energy_list = []
        self.normalization_list = []
        for element in self.materials:
            A = element[0]
            M_N = A*self.M_P  # nucleon mass [keV/c^2]
            F_C = (1.23*A**(1./3.)-.6)*1e-15/self.METERS_TO_EV*1e3  # factor for nuclear radius r_0 [c*hbar/keV]
            m_chi = np.copy(pars)*1e6  # DM mass [keV/c^2]
            mu_p = self.M_P*m_chi/(self.M_P+m_chi)  # reduced mass with proton mass [keV/c^2]
            mu_N = M_N*m_chi/(M_N+m_chi)  # reduced mass with nucleon mass [keV/c^2]
            z = np.sqrt(3/2)*self.V_ESC/self.W  # factor needed for normalization
            n = erf(z) - 2/np.sqrt(np.pi)*z*np.exp(-z**2)  # normalization factor for the analytical solution of the integral
            r_0 = np.sqrt((F_C**2)+7./3.*(np.pi**2)*(self.F_A**2)-5.*(self.F_S**2))  # nuclear radius [c*hbar/keV]
            eta = np.sqrt(3/2)*self.V_EARTH/self.W  # factor needed for the analytical solution of the integral

            q = np.sqrt(2 * M_N * self.energy_grid)  # momentum transferred in the scattering process [keV/c])
            spherical_bessel_j1 = (np.sin(q*r_0)/(q*r_0)**2)-(np.cos(q*r_0)/(q*r_0))
            f = 3.*spherical_bessel_j1/(q*r_0)*np.exp(-0.5*q*q*self.S*self.S)
            v_min = np.sqrt(self.energy_grid * M_N / (2 * mu_N ** 2))  # lowest speed of WIMP that can induce a nuclear recoil of E_R [c]
            x_min = np.sqrt(3*v_min**2/(2*self.W**2))  # factor needed for the analytical solution of the integral
            x_min_1, x_min_2, x_min_3 = [], [], []
            for item in x_min:
                if item < (z-eta):
                    x_min_1.append(item)
                elif (z-eta) <= item < (z+eta):
                    x_min_2.append(item)
                else:
                    x_min_3.append(item)
            integral_1 = np.sqrt(np.pi)/2*(erf(x_min_1+eta)-erf(x_min_1-eta))-2*eta*np.exp(-z**2)
            integral_2 = np.sqrt(np.pi)/2*(erf(z)-erf(x_min_2-eta))-np.exp(-z**2)*(z+eta-x_min_2)
            integral_3 = np.zeros(len(x_min_3))  # TODO: Je kleiner die Masse, desto mehr Werte in xmin3 und umgekehrt -> true division error
            integral = np.concatenate((integral_1, integral_2, integral_3), axis=None)
            integral *= 1/(n*eta)*np.sqrt(3/(2*np.pi*self.W**2))
            rates_per_energy = self.RHO_X/(2.*mu_p**2*m_chi)*A**2*f**2*integral  # total interaction rate R per energy E [c^2/(hbar*keV)] per sigma [pbarn]
            rates_per_energy *= self.DIMENSION_FACTOR
            rates_per_energy = rates_per_energy*self.cut_eff
            normalization = integrate.trapz(rates_per_energy, self.energy_grid)  # normalization factor of the pdf
            rates_per_energy = rates_per_energy/normalization  # normalized pdf
            rates_per_energy_list.append(rates_per_energy)
            self.normalization_list.append(normalization)

        return self.energy_grid, rates_per_energy_list

    def cdf(self, pars: float):
        """
        Calculate the cumulative density function of the signal model, evaluated at the set energy grid.

        :param pars: The mass of the dark matter particle.
        :return: The energy grid and the cumulative density functions evaluated on the grid.
        :rtype: list
        """
        m_chi = np.copy(pars)*1e6  # DM mass [keV/c^2]
        pdf = SignalModel.pdf(self, self.energy_grid, m_chi)
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

    @staticmethod
    def cdf_from_convoluted_pdf(pdf_sum_convoluted: list):
        """
        Calculate the cumulative density function of the signal model calculated from the convoluted probability density
        function evaluated at the set energy grid.
        :param pdf_sum_convoluted: The energy grid and the summed and convoluted probability density function.
        :return: The energy grid and the cumulative density function evaluated on the grid.
        :rtype: list
        """
        cdf = np.zeros(np.shape(pdf_sum_convoluted[0])[0])
        for i in range(np.shape(pdf_sum_convoluted[0])[0]-1):
            cdf[i+1] = cdf[i]+integrate.trapz(pdf_sum_convoluted[1][i:i+2], pdf_sum_convoluted[0][i:i+2])
        return pdf_sum_convoluted[0], cdf

    def rvs(self, size: int, pars: float):  # TODO: naming
        """
        Draw a sample from the signal model. Inputs are the size and the mass of the dark matter.

        :param size: The sample size
        :param pars: The mass of the dark matter particle.
        :return: The recoil energies of the drawn sample.
        :rtype: numpy array
        """
        cdf = SignalModel.cdf(self, pars*1e6)
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

    def rvs2(self, size: int, cdf: list):  # TODO: naming
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

    @staticmethod
    def rvs_sum(size: int, cdf_sum: list):
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

    @staticmethod
    def pdf_sum(pdf: list, materials: list):
        """
        Sum the pdfs of the given materials according to their percentage in the molecule.

        :param pdf: The energy grid and the probability density functions.
        :param materials: List of materials with the information: mass, percentile mass in the molecule, element name
        :return: Energy grid and summed probability density function.
        :rtype: list
        """

        pdf_sum = np.zeros(np.shape(pdf[0])[0])
        for i in range(len(pdf[1])):
            pdf_sum += pdf[1][i]*materials[i][1]

        return pdf[0], pdf_sum

    @staticmethod
    def cdf_sum(cdf: list, materials: list):
        """
        Sum the cdfs of the given materials according to their percentage in the molecule.

        :param cdf: The energy grid and the probability density functions.
        :param materials: List of materials with the information: mass, percentile mass in the molecule, element name
        :return: Energy grid and summed cumulative density function.
        :rtype: list
        """

        cdf_sum = np.zeros(np.shape(cdf[0])[0])
        for i in range(len(cdf[1])):
            cdf_sum += cdf[1][i]*materials[i][1]

        return cdf[0], cdf_sum

    def get_mus(self, pdf_sum_convoluted: list, sigmas: list):
        """
        Calculate the expected amount of events μ.

        :param pdf_sum_convoluted: The summed and convoluted probability density function.
        :param sigmas: Cross-sections [pbarn].
        :return: Expected amount of events μ.
        :rtype: numpy array
        """
        pdf_sum = pdf_sum_convoluted[1]*self.cut_eff
        self.normalization_factor = sum([self.normalization_list[i]*self.materials[i][1] for i in range(len(self.materials))])
        pdf_sum_limited_x = pdf_sum_convoluted[0][np.where(pdf_sum_convoluted[0] > self.threshold)[0][0]:np.where(pdf_sum_convoluted[0] < self.upper_integral_limit)[0][-1]+1]
        pdf_sum_limited_y = pdf_sum[np.where(pdf_sum_convoluted[0] > self.threshold)[0][0]:np.where(pdf_sum_convoluted[0] < self.upper_integral_limit)[0][-1]+1]
        mus = integrate.trapz(pdf_sum_limited_y, pdf_sum_limited_x)*np.array(sigmas)*self.normalization_factor*self.exposure
        return mus

    def set_cut_eff(self, file_name):
        """
        Set cut efficiency.

        :param file_name: File path and name.
        :param plot:
        :return: None
        """
        y_interp = []
        with open(file_name, 'r', encoding='UTF8', newline='') as f:
            dataset = f.readlines()
            dataset = [line.strip('\n') for line in dataset if line[0] != '#']
            dataset = [line.split('\t') for line in dataset if line[0] != '#']
        self.cut_eff.append([float(number[0]) for number in dataset])
        self.cut_eff.append([float(number[1]) for number in dataset])
        y_interp = np.interp(self.energy_grid, self.cut_eff[0], self.cut_eff[1])
        self.cut_eff = np.copy(y_interp)
        return

    def pdf_sum_convolution(self, pdf_sum):
        """
        Calculate the convolution of the summed pdf in order to take into account the finite energy resolution.

        :param pdf_sum: The summed probability density function.
        :return: Energy grid and the summed and convoluted pdf.
        :rtype: list
        """
        gauss_x = np.arange(-10*self.resolution, 10*self.resolution, 0.0005)  # 10 sigma Bereich, 0.5 eV steps
        gauss = norm.pdf(gauss_x, scale=self.resolution)
        pdf_sum_convoluted = np.convolve(pdf_sum[1], gauss, mode='same')/np.sum(gauss)
        return self.energy_grid, pdf_sum_convoluted

    # m_N_min is the nucleus with the smallest mass (for NaI its Na)  # TODO: Include?
    def _e_max_func(self, m_dm, materials):
        m_N_min = min([materials[i][0] for i in range(len(materials))])*self.M_P*1e-6
        mu_min = (m_dm * m_N_min)/(m_dm + m_N_min)
        e_max = (2 * mu_min**2 * (self.V_EARTH*const.c*1e-3+self.V_ESC*const.c*1e-3)**2)/m_N_min
        conversion_factor = 10**6/(const.c*1e-3)**2
        return e_max * conversion_factor

    # minimum mass we draw limits for is that one which has a E_max value <= threshold - 2*resolution
    def _find_dm_min(self, materials):

        def _to_find(m_dm):
            return self._e_max_func(m_dm, materials)-(self.threshold-2*self.resolution)

        result = optimize.root(_to_find, 1)
        return result.x[0]

import random
import numpy as np
from scipy import integrate
from scipy import constants as const
from scipy.special import erf
from scipy.special import spherical_jn
import matplotlib.pyplot as plt


class SignalModel:
    """
    A class that ...
    """

    def __init__(self):  #, A):  # def __init__(self):
        # Here global parameters for the signal model can be fixed. E.g. the escape velocity, ...
        # self.A = A
        self.METERS_TO_EV = const.hbar*const.c/const.e  # conversion factor to convert meters in c*hbar/eV
        self.DIMENSION_FACTOR = const.hbar/86400  # conversion factor to convert the dimension of dR/dE from c^2/(hbar*keV) in 1/(kg*d*keV)
        self.RHO_X = 0.3e6*1e6*self.METERS_TO_EV**3/1e9  # density [keV/c^2/(c*hbar/keV)^3]
        self.M_P = const.physical_constants["proton mass energy equivalent in MeV"][0]*10**3  # proton mass [keV/c^2]
        self.S = 1e-15/self.METERS_TO_EV*1e3  # skin thickness [c*hbar/keV]
        # self.SIGMA_P = np.pi * (const.physical_constants["proton rms charge radius"][0]/self.METERS_TO_EV*10**3)**2  # proton rms charge cross section [(c*hbar/keV)^2] # 10**(-8)*2.56*10**(-9)*10**(-12)
        self.SIGMA_P = 10**(-8)*2.56*10**(-9)*10**(-12)
        self.V_ESC = 554000./const.c  # escape velocity [c]
        self.W = 270000./const.c  # root mean square velocity of the DM particles [c]
        self.V_EARTH = 220*1.05e3/const.c  # earth velocity [c]
        self.F_A = .52e-15/self.METERS_TO_EV*1e3  # factor for nuclear radius r_0 [c*hbar/keV]
        self.F_S = .9e-15/self.METERS_TO_EV*1e3  # factor for nuclear radius r_0 [c*hbar/keV]
        self.exposure = 0.
        self.resolution = 0.
        self.threshold = 0.
        self.materials = []
        pass

    def set_detector(self, exposure: float, resolution: float, threshold: float, materials: list, upper_integral_limit: float):
        """

        :param exposure:
        :param resolution:
        :param threshold:
        :param materials:
        :return:
        """

        self.exposure = exposure
        self.resolution = resolution
        self.threshold = threshold
        self.materials = materials
        self.upper_integral_limit = upper_integral_limit

    def pdf(self, x, pars):
        """
        The probability density function of the signal model, evaluated at a given grid.

        :param x: The grid for the evaluation.
        :type x: list
        :param pars: The signal parameters.
        :type pars: list
        :return: The evaluated probability density function on the grid.
        :rtype: list
        """
        rates_per_energy_list = []
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
            recoil_energies = np.copy(x)  # np.arange(0.01, 200, 0.01)  # recoil energies [keV]
            eta = np.sqrt(3/2)*self.V_EARTH/self.W  # factor needed for the analytical solution of the integral

            q = np.sqrt(2*M_N*recoil_energies)  # momentum transferred in the scattering process [keV/c]
            # f = 3*spherical_jn(1, q*r_0)/(q*r_0)*np.exp(-0.5*q*q*self.S*self.S)  # TODO: DELETE?
            spherical_bessel_j1 = (np.sin(q*r_0)/(q*r_0)**2)-(np.cos(q*r_0)/(q*r_0))  # TODO: DELETE?
            f = 3.*spherical_bessel_j1/(q*r_0)*np.exp(-0.5*q*q*self.S*self.S)  # TODO: DELETE?
            v_min = np.sqrt(recoil_energies*M_N/(2*mu_N**2))  # lowest speed of WIMP that can induce a nuclear recoil of E_R [c]
            x_min = np.sqrt(3*v_min**2/(2*self.W**2))  # factor needed for the analytical solution of the integral
            x_min_1 = [item for item in x_min if item < (z-eta)]
            x_min_2 = [item for item in x_min if (z-eta) < item < (z+eta)]
            x_min_3 = [item for item in x_min if (z+eta) < item]
            integral_1 = np.sqrt(np.pi)/2*(erf(x_min_1+eta)-erf(x_min_1-eta))-2*eta*np.exp(-z**2)
            integral_2 = np.sqrt(np.pi)/2*(erf(z)-erf(x_min_2-eta))-np.exp(-z**2)*(z+eta-x_min_2)
            integral_3 = np.zeros(len(x_min_3))  # TODO: Je kleiner die Masse, desto mehr Werte in xmin3 und umgekehrt -> true division error
            integral = np.concatenate((integral_1, integral_2, integral_3), axis=None)
            integral *= 1/(n*eta)*np.sqrt(3/(2*np.pi*self.W**2))
            rates_per_energy = self.RHO_X/(2.*mu_p**2*m_chi)*A**2*f**2*integral  # total interaction rate R per energy E [c^2/(hbar*keV)] per sigma [pbarn]
            rates_per_energy /= self.DIMENSION_FACTOR
            # TODO: dimension analysis
            normalization = integrate.trapz(rates_per_energy, recoil_energies)  # normalization factor of the pdf
            rates_per_energy = rates_per_energy/normalization  # normalized pdf
            rates_per_energy_list.append(rates_per_energy)

        return recoil_energies, rates_per_energy_list

    def cdf(self, x, pars):
        """
        The cummulative density function of the signal model, evaluated at a given grid.

        :param x: The grid for the evaluation.
        :type x: list
        :param pars: The signal parameters.
        :type pars: list
        :return: The evaluated cummulative density function on the grid.
        :rtype: list
        """
        m_chi = np.copy(pars)*1e6  # DM mass [keV/c^2]
        recoil_energies = np.copy(x)
        pdf = SignalModel.pdf(self, recoil_energies, m_chi)
        cdf = np.zeros((np.shape(self.materials)[0], np.shape(pdf[0])[0]))
        for j in range(len(self.materials)):  # TODO: np.shape
            for i in range(np.shape(pdf[0])[0]-1):
                cdf[j][i+1] = cdf[j][i]+integrate.trapz(pdf[1][j][i:i+2], pdf[0][i:i+2])
        return pdf[0], cdf

    def cdf2(self, pdf):
        """
        The cummulative density function of the signal model, evaluated at a given grid.

        :param pdf:
        :type pdf:
        :return: The evaluated cummulative density function on the grid.
        :rtype: list
        """
        cdf = np.zeros((np.shape(self.materials)[0], np.shape(pdf[0])[0]))
        for j in range(len(self.materials)):
            for i in range(np.shape(pdf[0])[0]-1):
                cdf[j][i+1] = cdf[j][i]+integrate.trapz(pdf[1][j][i:i+2], pdf[0][i:i+2])
        return pdf[0], cdf

    def rvs(self, size, x, pars):
        """
        Draw a sample from the signal model.

        :param x:
        :type x:
        :param size: The sample size
        :type size: int
        :param pars: The signal parameters.
        :type pars: list
        :return: The recoil energies (and possible other parameters) of the drawn sample.
        :rtype: list
        """
        cdf = SignalModel.cdf(self, x, pars*1e6)
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

    def rvs2(self, size, cdf):
        """
        Draw a sample from the signal model.

        :param cdf:
        :type cdf:
        :param size: The sample size
        :type size: int
        :return: The recoil energies (and possible other parameters) of the drawn sample.
        :rtype: list
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

    def rvs_sum(self, size, cdf):
        """
        Draw a sample from the signal model.

        :param cdf:
        :type cdf:
        :param size: The sample size
        :type size: int
        :return: The recoil energies (and possible other parameters) of the drawn sample.
        :rtype: list
        """
        sample = []
        for i in range(size):
            rand = random.random()
            for j in range(len(cdf[0])):
                if cdf[1][j] >= rand:
                    sample.append(cdf[0][j-1])
                    break
        return np.array(sample)

    def pdf_sum(self, pdf, materials):
        """

        :param pdf:
        :type pdf:
        :param materials:
        :type materials:
        :return:
        """
        """pdf_sum = np.zeros(np.shape(pdf[0])[0])
        for i in range(len(pdf[0])):
            for j in range(len(pdf[1])):
                pdf_sum[i] += pdf[1][j][i]*materials[j][1]"""

        pdf_sum_2 = np.zeros(np.shape(pdf[0])[0])
        for i in range(len(pdf[1])):
            pdf_sum_2 += pdf[1][i]*materials[i][1]

        """pdf_sum_3 = sum([pdf[1][i]*materials[i][1] for i in range(len(pdf[1]))])"""
        return pdf[0], pdf_sum_2

    def cdf_sum(self, cdf, materials):
        """

        :param cdf:
        :type cdf:
        :param materials:
        :type materials:
        :return:
        """
        """cdf_sum = np.zeros(np.shape(pdf[0])[0])
        for i in range(len(cdf[0])):
            for j in range(len(cdf[1])):
                cdf_sum[i] += cdf[1][j][i]*materials[j][1]"""

        cdf_sum_2 = np.zeros(np.shape(cdf[0])[0])
        for i in range(len(cdf[1])):
            cdf_sum_2 += cdf[1][i]*materials[i][1]

        """cdf_sum_3 = sum([cdf[1][i]*materials[i][1] for i in range(len(cdf[1]))])"""
        return cdf[0], cdf_sum_2

    def log_plot(self, pdf, pdf_sum, materials):
        """

        :param pdf:
        :type: pdf:
        :param pdf_sum:
        :type pdf_sum:
        :param materials:
        :type materials:
        :return:
        """
        plt.yscale("log")
        plt.xlim(0, 40)
        # plt.ylim(10**(-12), 10**0)  # TODO: limits aktivieren
        for material in range(len(materials)):
            label = 'ratio_{0}*log_pdf_{1}'.format(materials[material][1], materials[material][2])
            plt.plot(pdf[0], materials[material][1]*pdf[1][material], label=label)
        plt.plot(pdf_sum[0], pdf_sum[1], label='log_pdf_sum')
        plt.legend()
        plt.show()
        return

    def linear_sum_plot(self, pdf_sum, cdf_sum, samples_sum):
        """

        :param pdf_sum:
        :type pdf_sum:
        :param cdf_sum:
        :type cdf_sum:
        :param samples_sum:
        :type samples_sum:
        :return:
        """
        plt.yscale("linear")
        plt.xlim(0, 40)
        # plt.ylim(0, 1)  # TODO: limits aktivieren
        plt.plot(pdf_sum[0], pdf_sum[1], label='lin_PDF_sum')
        plt.plot(cdf_sum[0], cdf_sum[1], label='CDF_sum')
        plt.hist(samples_sum, 500, density=True, label='Histogram')
        plt.legend()
        plt.show()
        return

    def get_mus(self, pdf_sum: list, sigmas: list):
        """

        :param pdf_sum:
        :type pdf_sum:
        :param sigmas:
        :type sigmas:
        :return:
        """
        pdf_sum_limited_x = pdf_sum[0][:np.where(pdf_sum[0] < self.upper_integral_limit)[0][-1]+1]
        pdf_sum_limited_y = pdf_sum[1][:len(pdf_sum_limited_x)]
        mus = sum(pdf_sum_limited_x*pdf_sum_limited_y)*np.array(sigmas)
        return mus

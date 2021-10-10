import random
import numpy as np
from scipy import integrate
from scipy import constants as const
from scipy.special import erf
from scipy.special import spherical_jn


class SignalModel:

    def __init__(self, A):  # def __init__(self):
        # Here global parameters for the signal model can be fixed. E.g. the escape velocity, ...
        self.A = A
        self.METERS_TO_EV = const.hbar * const.c / const.e  # conversion factor to convert meters in c*hbar/eV
        self.RHO_X = 0.3 * 10 ** 6 * 10 ** 6 * self.METERS_TO_EV ** 3 / 10 ** 9  # density [keV/c^2/(c*hbar/keV)^3]
        self.M_P = const.physical_constants["proton mass energy equivalent in MeV"][0]*10**3  # proton mass [keV/c^2]
        self.M_N = self.A*self.M_P  # nucleon mass [keV/c^2]
        self.S = 10 ** (-15) / self.METERS_TO_EV * 10 ** 3  # skin thickness [c*hbar/keV]
        self.SIGMA_P = np.pi * (const.physical_constants["proton rms charge radius"][0] / self.METERS_TO_EV * 10 ** 3) ** 2  # proton rms charge cross section [(c*hbar/keV)^2]
        self.V_ESC = 554000./const.c  # escape velocity [c]
        self.W = 270000./const.c  # root mean square velocity of the DM particles [c]
        self.V_EARTH = 220*1.05*10**3/const.c  # earth velocity [c]
        self.F_A = .52 * 10 ** (-15) / self.METERS_TO_EV * 10 ** 3  # factor for nuclear radius r_0 [c*hbar/keV]
        self.F_S = .9 * 10 ** (-15) / self.METERS_TO_EV * 10 ** 3  # factor for nuclear radius r_0 [c*hbar/keV]
        self.F_C = (1.23*self.A**(1./3.)-.6) * 10 ** (-15) / self.METERS_TO_EV * 10 ** 3  # factor for nuclear radius r_0 [c*hbar/keV]
        pass

    def set_detector(self, exposure: float, resolution: float, threshold: float, material: int):

        self.exposure = exposure
        self.resolution = resolution
        self.threshold = threshold
        self.A = material  # TODO: material in init or set_detector? -> calculations with A

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
        m_chi = np.copy(pars)*10**6  # 10.*10**6  # DM mass [keV/c^2]
        mu_p = self.M_P*m_chi/(self.M_P+m_chi)  # reduced mass with proton mass [keV/c^2]
        mu_N = self.M_N*m_chi/(self.M_N+m_chi)  # reduced mass with nucleon mass [keV/c^2]
        z = np.sqrt(3/2)*self.V_ESC/self.W  # factor needed for normalization
        n = erf(z) - 2/np.sqrt(np.pi)*z*np.exp(-z**2)  # normalization factor for the analytical solution of the integral
        r_0 = np.sqrt((self.F_C**2)+7./3.*(np.pi**2)*(self.F_A**2)-5.*(self.F_S**2))  # nuclear radius [c*hbar/keV]
        recoil_energies = np.copy(x)  # np.arange(0.01, 200, 0.01)  # recoil energies [keV] TODO: x or recoil energies as argument?
        eta = np.sqrt(3/2)*self.V_EARTH/self.W  # factor needed for the analytical solution of the integral

        q = np.sqrt(2*self.M_N*recoil_energies)  # momentum transferred in the scattering process [keV/c] TODO: DIMENSION CORRECT?
        f = 3*spherical_jn(1, q*r_0)/(q*r_0)*np.exp(-0.5*q*q*self.S*self.S)  # TODO: DIMENSIONSLOS?
        #spherical_bessel_j1 = (np.sin(q*r_0)/(q*r_0)**2)-(np.cos(q*r_0)/(q*r_0)) TODO: DELETE?
        #f = 3.*spherical_bessel_j1/(q*r_0)*np.exp(-0.5*q*q*S*S) TODO: DELETE?
        v_min = np.sqrt(recoil_energies*self.M_N/(2*mu_N**2))  # lowest speed of WIMP that can induce a nuclear recoil of E_R [c]
        x_min = np.sqrt(3*v_min**2/(2*self.W**2))  # factor needed for the analytical solution of the integral
        integral = 1/(n*eta)*(3/np.sqrt(2*np.pi*self.W**2)*(np.sqrt(np.pi)/2*(erf(x_min-eta)-erf(x_min+eta))-2*eta*np.exp(-z**2)))  # analytical solution of the integral
        rates_per_energy = self.RHO_X/(2.*mu_p**2*m_chi)*self.A**2*self.SIGMA_P*f**2*integral  # total interaction rate R per energy E [1/hbar] TODO: DIMENSION CORRECT?
        normalization = integrate.trapz(rates_per_energy, recoil_energies)  # normalization factor of the pdf
        rates_per_energy = rates_per_energy/normalization  # normalized pdf

        # The part below is needed, if one wants to plot only until the first dip
        rates_per_energy_till_first_dip = []
        for i in range(len(rates_per_energy)):
            if i < len(rates_per_energy)-1:
                if rates_per_energy[i+1]-rates_per_energy[i] < 0:
                    rates_per_energy_till_first_dip.append(rates_per_energy[i])
                else:
                    recoil_energies = recoil_energies[:i]
                    break
            else:
                recoil_energies = recoil_energies[:i]
                break
        rates_per_energy_till_first_dip = np.array(rates_per_energy_till_first_dip)
        return recoil_energies, rates_per_energy_till_first_dip

        # return recoil_energies, rates_per_energy

    def cdf(self, x, pars):  # TODO: DECIDE WHICH METHOD TO USE
        """
        The cummulative density function of the signal model, evaluated at a given grid.

        :param x: The grid for the evaluation.
        :type x: list
        :param pars: The signal parameters.
        :type pars: list
        :return: The evaluated cummulative density function on the grid.
        :rtype: list
        """
        m_chi = np.copy(pars)*10**6  # DM mass [keV/c^2]
        recoil_energies = np.copy(x)  # np.arange(0.01, 200, 0.01)  # recoil energies [keV]
        pdf = SignalModel.pdf(self, recoil_energies, m_chi)
        cdf = np.zeros(np.shape(pdf[0])[0])
        for i in range(np.shape(pdf[0])[0]-1):
            cdf[i+1] = cdf[i]+integrate.trapz(pdf[1][i:i+2], pdf[0][i:i+2])
        return pdf[0], cdf

    def cdf2(self, pdf):  # TODO: DECIDE WHICH METHOD TO USE
        """
        The cummulative density function of the signal model, evaluated at a given grid.

        :param x: The grid for the evaluation.
        :type x: list
        :param pars: The signal parameters.
        :type pars: list
        :return: The evaluated cummulative density function on the grid.
        :rtype: list
        """
        cdf = np.zeros(np.shape(pdf[0])[0])
        for i in range(np.shape(pdf[0])[0]-1):
            cdf[i+1] = cdf[i]+integrate.trapz(pdf[1][i:i+2], pdf[0][i:i+2])
        return pdf[0], cdf

    def rvs(self, size, x, pars):
        """
        Draw a sample from the signal model.

        :param size: The sample size
        :type size: int
        :param pars: The signal parameters.
        :type pars: list
        :return: The recoil energies (and possible other parameters) of the drawn sample.
        :rtype: list
        """
        cdf = SignalModel.cdf(self, x, pars)
        sample = []
        for i in range(size):
            rand = random.random()
            for j in range(len(cdf[0])):
                if cdf[1][j] >= rand:
                    sample.append(cdf[0][j-1])
                    break
        return np.array(sample)

    def rvs2(self, size, cdf):  #  pars):
        """
        Draw a sample from the signal model.

        :param size: The sample size
        :type size: int
        :param pars: The signal parameters.
        :type pars: list
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

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import constants as const
from scipy.special import erf
from scipy.special import spherical_jn

# PARAMETER #

METERS_TO_EV = const.hbar*const.c/const.e  # conversion factor to convert meters in c*hbar/eV
RHO_X = 0.3*10**6*10**6*METERS_TO_EV**3/10**9  # density [keV/c^2/(c*hbar/keV)^3]
A = 184.  # 184 mass number of tungsten, 131 mass number of xenon
M_P = const.physical_constants["proton mass energy equivalent in MeV"][0]*10**3  # proton mass [keV/c^2]
M_N = A*M_P  # nucleon mass [keV/c^2]
S = 10**(-15)/METERS_TO_EV*10**3  # skin thickness [c*hbar/keV]
SIGMA_P = np.pi*(const.physical_constants["proton rms charge radius"][0]/METERS_TO_EV*10**3)**2  # proton rms charge cross section [(c*hbar/keV)^2]
V_ESC = 554000./const.c  # escape velocity [c]
W = 270000./const.c  # root mean square velocity of the DM particles [c]
V_EARTH = 220*1.05*10**3/const.c  # earth velocity [c]
F_A = .52*10**(-15)/METERS_TO_EV*10**3  # factor for nuclear radius r_0 [c*hbar/keV]
F_S = .9*10**(-15)/METERS_TO_EV*10**3  # factor for nuclear radius r_0 [c*hbar/keV]
F_C = (1.23*A**(1./3.)-.6)*10**(-15)/METERS_TO_EV*10**3  # factor for nuclear radius r_0 [c*hbar/keV]


def pdf():
    m_chi = 10.*10**6  # DM mass [keV/c^2]
    mu_p = M_P*m_chi/(M_P+m_chi)  # reduced mass with proton mass [keV/c^2]
    mu_N = M_N*m_chi/(M_N+m_chi)  # reduced mass with nucleon mass [keV/c^2]
    z = np.sqrt(3/2)*V_ESC/W  # factor needed for normalization
    n = erf(z) - 2/np.sqrt(np.pi)*z*np.exp(-z**2)  # normalization factor for the analytical solution of the integral
    # r_nuclear = 1.2*A**(1/3)*10**(-15)/METERS_TO_EV*10**3  # nuclear radius [1/keV] TODO: DELETE?
    # r_0 = np.sqrt(r_nuclear**2 - 5*S**2)  # [1/keV] TODO: DELETE?
    r_0 = np.sqrt((F_C**2)+7./3.*(np.pi**2)*(F_A**2)-5.*(F_S**2))  # nuclear radius [c*hbar/keV]
    recoil_energies = np.arange(0.01, 200, 0.01)  # recoil energies [keV]
    eta = np.sqrt(3/2)*V_EARTH/W  # factor needed for the analytical solution of the integral

    q = np.sqrt(2*M_N*recoil_energies)  # momentum transferred in the scattering process [keV/c] TODO: DIMENSION CORRECT?
    f = 3*spherical_jn(1, q*r_0)/(q*r_0)*np.exp(-0.5*q*q*S*S)  # TODO: DIMENSIONSLOS?
    plt.yscale("log")
    plt.plot(recoil_energies, f*f, label="f**2")
    plt.ylim([10**(-6), 1])
    plt.legend()
    plt.show()
    #spherical_bessel_j1 = (np.sin(q*r_0)/(q*r_0)**2)-(np.cos(q*r_0)/(q*r_0)) TODO: DELETE?
    #f = 3.*spherical_bessel_j1/(q*r_0)*np.exp(-0.5*q*q*S*S) TODO: DELETE?
    v_min = np.sqrt(recoil_energies*M_N/(2*mu_N**2))  # lowest speed of WIMP that can induce a nuclear recoil of E_R [c]
    x_min = np.sqrt(3*v_min**2/(2*W**2))  # factor needed for the analytical solution of the integral
    integral = 1/(n*eta)*(3/np.sqrt(2*np.pi*W**2)*(np.sqrt(np.pi)/2*(erf(x_min-eta)-erf(x_min+eta))-2*eta*np.exp(-z**2)))  # analytical solution of the integral
    rates_per_energy = RHO_X/(2.*mu_p**2*m_chi)*A**2*SIGMA_P*f**2*integral  # total interaction rate R per energy E [1/hbar] TODO: DIMENSION CORRECT?
    normalization = integrate.trapz(rates_per_energy, recoil_energies)  # normalization factor of the pdf
    rates_per_energy = rates_per_energy/normalization  # normalized pdf
    return recoil_energies, rates_per_energy


def cdf(pdf):
    cdf = [integrate.trapz(pdf[1][:i+1], pdf[0][:i+1]) for i in range(np.shape(pdf[0])[0])]  # cumulative density function
    return pdf[0], cdf


def rvs(n, cdf):
    sample = []
    for i in range(n):
        rand = random.random()
        for j in range(len(cdf[0])):
            if cdf[1][j] >= rand:
                sample.append(cdf[0][j-1])
                break
    return np.array(sample)


pdf = pdf()
cdf = cdf(pdf)
samples = rvs(10000, cdf)
plt.yscale("log")
plt.plot(pdf[0], pdf[1], label="log_PDF")
plt.legend()
plt.show()
plt.yscale("linear")
plt.plot(pdf[0], pdf[1], label="lin_PDF")
plt.plot(cdf[0], cdf[1], label="CDF")
plt.hist(samples, 100, density=True, label="Histogram")
plt.legend()
plt.show()
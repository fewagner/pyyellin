import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import constants as const
from scipy.special import erf
from scipy.special import spherical_jn

# PARAMETER #

METERS_TO_EV = const.hbar*const.c/const.e
RHO_X = 0.3  # density [GeV/c^2/cm^3]
A = 184.  # mass number of tungsten
M_N = A*const.physical_constants["proton mass energy equivalent in MeV"][0]/10**3  # proton mass [GeV/c^2]
S = 1.  # skin thickness [fm]
SIGMA_P = np.pi*(const.physical_constants["proton rms charge radius"][0])*10**30  # proton rms charge cross section [fm^2]
V_ESC = 554.  # escape velocity [km/s]
W = 270.  # root mean square velocity of the DM particles [km/s]
V_EARTH = 220*1.05  # average earth velocity through the galaxy [km/s]
F_A = .52  # in fm
F_S = .9  # in fm
F_C = 1.23 * A ** (1./3.) - .6  # in fm


def pdf():
    m_chi = 10.  # DM mass [GeV]
    mu_p = M_N*m_chi / (M_N+m_chi)  # reduced mass [GeV]
    z = np.sqrt(3./2.)*V_ESC/W  # []
    eta = np.sqrt(3./2.)*V_EARTH/W  # []
    n = erf(z) - 2./np.sqrt(np.pi)*z*np.exp(-z**2)  # []
    # r_nuclear = 1.2*A**(1./3.)  # nuclear radius [fm]
    # r_0 = np.sqrt(r_nuclear**2 - 5*S**2)  # [fm]
    r_0 = np.sqrt((F_C**2)+7./3.*(np.pi**2)*(F_A**2)-5.*(F_S**2))
    recoil_energies = np.arange(0.01, 200, 0.01)  # recoil energies [keV]

    q = np.sqrt(2.*M_N*recoil_energies) / 197.326960
    f = 3.*spherical_jn(1, q*r_0)/(q*r_0)*np.exp(-0.5*q*q*S*S)
    plt.yscale("log")
    plt.plot(recoil_energies, f*f, label="f**2")
    plt.ylim([10**(-6), 1])
    plt.legend()
    plt.show()
    #spherical_bessel_j1 = (np.sin(q*r_0)/(q*r_0)**2)-(np.cos(q*r_0)/(q*r_0))
    #f = 3.*spherical_bessel_j1/(q*r_0)*np.exp(-0.5*q*q*S*S)
    v_min = np.sqrt(recoil_energies * 10 ** (-6) * M_N / (2 * mu_p ** 2))  # 10^-6 for conversion of keV to GeV
    x_min = np.sqrt(3*v_min**2*(const.c**2/10**6)/(2*W**2))  # 10**6 for conversion of m/s to km/s, c^2 for conversion of km/s to units of c
    integral = 1/(n*eta)*(np.sqrt(3/(2*np.pi*W**2))*(np.sqrt(np.pi)/2*(erf(x_min-eta)-erf(x_min+eta))-2*eta*np.exp(-z**2)))
    rates_per_energy = RHO_X/(2.*mu_p**2*m_chi)*A**2*SIGMA_P*f**2*integral
    normalization = integrate.trapz(rates_per_energy, recoil_energies)
    rates_per_energy = rates_per_energy/normalization
    return recoil_energies, rates_per_energy


def cdf(pdf):
    cdf = [integrate.trapz(pdf[1][:i+1], pdf[0][:i+1]) for i in range(np.shape(pdf[0])[0])]
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
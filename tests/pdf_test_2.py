import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import constants as const
from scipy.stats import norm
from scipy.special import erf
from scipy.special import spherical_jn


def pdf():
    rho_x = 0.3  # *10**9*const.e/const.c**2/10**(-6)  # density [GeV/c**2/cm**3]
    m_p = const.physical_constants["proton mass energy equivalent in MeV"][0]/10**3  # proton mass [GeV]
    m_chi = 10  # *10**9*const.e/(const.c**2)  # DM mass [GeV]
    mu_p = m_p*m_chi/(m_p+m_chi)  # reduced mass [GeV]
    a = 184  # *1.660499999628*10**(-27)
    mass_number = 184
    s = 1  # 10**(-15)  # skin thickness [fm]
    sigma_p = np.pi*const.physical_constants["proton rms charge radius"][0]**2*10**30  # proton rms charge cross section [m**2]
    v_esc = 650000  # escape velocity [m/s]
    w = 270000  # root mean square velocity of the DM particles [m/s]
    z = np.sqrt(3*v_esc**2/(2*w**2))
    n = erf(z) - 2/np.sqrt(np.pi)*z*np.exp(-z**2)
    r_nuclear = 1.2*mass_number**(1/3)  # *10**(-15)  # nuclear radius [fm]
    r_0 = np.sqrt(r_nuclear**2 - 5*s**2)  # [fm]
    recoil_energies = np.arange(1, 30, 0.01)  # *const.e)  # recoil energies [keV]
    v_earth = 220*1.12*10**3  # earth velocity [m/s]
    eta = np.sqrt(3*v_earth**2/(2*w**2))

    rates_per_energy = np.array([])
    list_f = np.array([])
    for recoil_energy in recoil_energies:
        q = np.sqrt(2*m_p*recoil_energy)
        #spherical_bessel_j1 = (np.sin(q*r_0)/(q*r_0)**2)-(np.cos(q*r_0)/(q*r_0))
        f = 3*spherical_jn(1, q*r_0)/(q*r_0)*np.exp(-0.5*q*q*s*s)
        #f = 3.*spherical_bessel_j1/(q*r_0)*np.exp(-0.5*q*q*s*s)
        v_min = np.sqrt(recoil_energy*m_p/(2*mu_p**2))  # *const.c
        #v_grid = np.arange(v_min, v_esc, 10)
        #integrand_y = np.array(list(map(lambda v: 1./n*(3./(2.*np.pi*w**2))**(3./2.)*np.exp(-3.*v**2/(2.*w**2))/v, v_grid)))
        #integral = integrate.trapz(integrand_y, v_grid)
        x_min = np.sqrt(3*v_min**2/(2*w**2))
        integral = 1/(n*eta)*(3/np.sqrt(2*np.pi*w**2)*(np.sqrt(np.pi)/2*(erf(x_min-eta)-erf(x_min+eta))-2*eta*np.exp(-z**2)))
        rate_per_energy = rho_x/(2.*mu_p**2*m_chi)*a**2*sigma_p*f**2*integral
        rates_per_energy = np.append(rates_per_energy, rate_per_energy)
        list_f = np.append(list_f, f*f)
    plt.plot(recoil_energies, list_f)
    plt.show()
    normalization = integrate.trapz(rates_per_energy, recoil_energies)
    rates_per_energy = rates_per_energy/normalization
    return recoil_energies, rates_per_energy


def pdf2():  # To test the cdf and rvs functions
    recoil_energies = np.arange(0, 100, .1)
    pdf = norm.pdf(recoil_energies, 50, 1)
    return recoil_energies, pdf


def cdf(pdf):
    cdf = np.array([])  # np.zeros schneller?
    for i in range(np.shape(pdf[0])[0]):
        cdf_value = integrate.trapz(pdf[1][:i+1], pdf[0][:i+1])
        cdf = np.append(cdf, cdf_value)
    return pdf[0], cdf


def cdf2(pdf):
    cdf2 = np.zeros(np.shape(pdf[0])[0])
    for i in range(np.shape(cdf2)[0]):
        cdf2[i] = integrate.trapz(pdf[1][:i+1], pdf[0][:i+1])
    return pdf[0], cdf2


def rvs(n, cdf):
    sample = np.array([])
    for i in range(n):
        rand = random.random()
        for j in range(np.shape(cdf[0])[0]):
            if cdf[1][j] >= rand:
                sample = np.append(sample, cdf[0][j-1])
                break
    return sample


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
plt.hist(samples, 100, density=True)
plt.legend()
plt.show()
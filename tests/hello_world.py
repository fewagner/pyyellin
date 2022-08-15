import numpy as np
import pyyellin as yell
import matplotlib.pyplot as plt

m_chi = 10.  # DM mass [GeV/c^2]
# recoil_energies = np.arange(0.01, 200.01, 0.01)  # recoil energies [keV]
sample_size = 10000
# recoil_energies = np.linspace(1, 200, 1)  # TODO: Stabilität
# print(recoil_energies)

materials = [[23., 23/150, 'Na'], [127., 127/150, 'I']]
sigmas = np.array([2, 3, 4, 5, 6])
sigmas = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
sigmas = np.linspace(1e-6, 1e-7, 100)
sigmas = np.arange(0, 101, 1)
signal = yell.SignalModel()
signal.set_energy_grid(np.arange(0.0005, 50.0005, 0.0005))
signal.set_detector(1, 1, 0, materials, 16)
signal.set_cut_eff('C3P1_DetA_cuteff.dat')
pdf = signal.pdf(m_chi)
cdf = signal.cdf_from_pdf(pdf)
samples = signal.rvs2(sample_size, cdf)
pdf_sum = signal.pdf_sum(pdf, materials)
cdf_sum = signal.cdf_sum(cdf, materials)
samples_sum = signal.rvs_sum(sample_size, cdf_sum)

mus = signal.get_mus(pdf_sum, sigmas)

with open('C:/Users/Fatih/Desktop/Project Minerva/Projektarbeit 1/xy_NaI.txt', 'r') as f:
    lines = f.readlines()[1:]
    line = [i.split() for i in lines]
    f.close()
energy = [float(i[0]) for i in line]
na = [float(i[1]) for i in line]
iod = [float(i[2]) for i in line]
nai = [float(i[3]) for i in line]

plt.yscale("log")
# plt.xlim(0, 40)
# plt.ylim(10**(-12), 10**0)
plt.plot(energy, na, label='na')
plt.plot(energy, iod, label='iod')
plt.plot(energy, nai, label='nai')

signal.log_plot(pdf, pdf_sum, materials)
signal.linear_sum_plot(pdf_sum, cdf_sum, samples_sum)

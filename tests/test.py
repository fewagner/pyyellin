import numpy as np
from matplotlib import pyplot as plt
import pyyellin as yell
from scipy.stats import percentileofscore

m_chi = 10.  # DM mass [GeV/c^2]
m_chi_list = [10., 9.]
recoil_energies = np.arange(0.001, 30.001, 0.001)  # recoil energies [keV]
sample_size = 10000
# recoil_energies = np.linspace(1, 200, 1)  # TODO: Stabilität
# print(recoil_energies)

materials = [[23., 23/150, 'Na'], [127., 127/150, 'I']]
sigmas = np.array([2, 3, 4, 5, 6])
signal = yell.SignalModel()
signal.set_detector(1, 1, 1, materials, 16e3)
pdf = signal.pdf(recoil_energies, m_chi_list)
cdf = signal.cdf2(pdf)
# samples = signal.rvs2(sample_size, cdf)
# pdf_sum = signal.pdf_sum(pdf, materials)
# cdf_sum = signal.cdf_sum(cdf, materials)
# samples_sum = signal.rvs_sum(sample_size, cdf_sum)

mus = signal.get_mus(pdf_sum, sigmas)

"""signal.log_plot(pdf, pdf_sum, materials)
signal.linear_sum_plot(pdf_sum, cdf_sum, samples_sum)"""


limit = yell.Limit()
list_test = np.arange(50, 100.5, 0.5)
list_test2 = np.arange(0.5, 10.5, 0.5)
limit.add_mus(list_test2)
limit.make_table(False, 100, 'table3')
limit.get_table('table3.csv')
# limit.get_data('C3P1_DetA_full.dat')
limit.get_data('C3P1_DetA_full_2.dat')
limit.get_limit(cdf_sum)

import numpy as np
import pyyellin as yell
import timeit
from matplotlib import pyplot as plt



with open('C3P1_DetA_DataRelease_SI.xy', 'r', encoding='UTF8', newline='') as f:
    dataset = f.readlines()
    dataset = [line.strip('\n') for line in dataset if line[0] != '#']
x_axis_plot = [float(number.split()[0]) for number in dataset]
# y_axis_plot = [float(number.split()[1]) for number in dataset]
# print('x_axis_plot =', x_axis_plot)
# print('y_axis_plot =', y_axis_plot)
# plt.plot(x_axis_plot, y_axis_plot, label='plot cresst 3')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.show()

test_list = [np.round(x*10**y, 3) for y in range(-2, 2) for x in range(1, 10)]
print(test_list)

start = timeit.default_timer()

m_chi = np.geomspace(1., 100., 25)
m_chi = list(m_chi)
for mass in x_axis_plot:
    m_chi.append(mass)
m_chi.sort()
m_chi = list(dict.fromkeys(m_chi))
m_chi = np.array(m_chi)
print(m_chi)
# m_chi = [10.]
# m_chi = np.linspace(3.5, 20, 10)
sample_size = 10000
# materials = [[23., 23/150, 'Na'], [127., 127/150, 'I']]
materials = [[40.078, 40.078/287.914, 'Ca'], [183.84, 183.84/287.914, 'W'], [15.999, 15.999*4/287.914, 'O']]

omega = yell.Ilok()
omega.set_detector(5.689, 0.0046, 0.0301, materials, 16.)
omega.set_cut_eff('C3P1_DetA_cuteff.dat')
omega.set_masses(m_chi)
# omega.set_sampling_size(sample_size)  # Optional, only needed for rvs functions
omega.set_sigma_interval(1e-6, 1e-2)
omega.set_mu_interval(0.25, 50)
# i_want_list = omega.calculate_items()
# print(i_want_list[0])
# table6 with omega.set_mu_interval(0.25, 250) and omega.set_table_variables(True, 2500, 'table6') -> 1.7 GB
omega.set_confidence_level(0.9)
omega.get_data('C3P1_DetA_AR.dat')
omega.set_table_variables(False, 100, 'table')
omega.get_limit()

stop = timeit.default_timer()
print('Time: ', stop - start)

# # print('thr = 0.0301 and res = 0.0046:')  # Optional
# # print('min DM mass in GeV = ', omega._find_dm_min(materials))
# # TODO: ist aber dieser dm_min nicht für WIMP Modell? Vielleict mass range einfach dem User überlassen ->
# #  andere Modells, und dann diese Funktion als zusätlicher Tool lassen, falls man die min Masse braucht im WIMP modell

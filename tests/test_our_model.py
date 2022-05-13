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
# print(test_list)

start = timeit.default_timer()

m_chi = np.geomspace(1., 100., 5)
m_chi = list(m_chi)
for mass in x_axis_plot:
    m_chi.append(mass)
m_chi.append(10.)
m_chi.sort()
m_chi = list(dict.fromkeys(m_chi))
m_chi = np.array(m_chi)
# print(m_chi)
sample_size = 10000
materials = [[40.078, 40.078/287.914, 'Ca', 'C3P1_DetA_eff_AR_Ca.dat'],
             [183.84, 183.84/287.914, 'W', 'C3P1_DetA_eff_AR_W.dat'],
             [15.999, 15.999*4/287.914, 'O', 'C3P1_DetA_eff_AR_O.dat']]
# materials = [[23., 23/150, 'Na'], [127., 127/150, 'I']]
# m_chi = [10.]

omega = yell.Ilok()
omega.set_detector(5.594, 0.0046, 0.0301, materials, 16.)
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
#
# print(i_want_list)
# print(i_want_list[list(m_chi).index(10.)])
#
# with open('C:/Users/Fatih/Desktop/Project Minerva/Projektarbeit 1/xy_NaI.txt', 'r') as f:
#     lines = f.readlines()[1:]
#     line = [i.split() for i in lines]
#     f.close()
# energy = [float(i[0]) for i in line]
# na = [float(i[1]) for i in line]
# iod = [float(i[2]) for i in line]
# nai = [float(i[3]) for i in line]
#
# plt.yscale("log")
# # plt.xlim(0, 40)
# # plt.ylim(10**(-12), 10**0)
# plt.plot(energy, na, label='na')
# plt.plot(energy, iod, label='iod')
# plt.plot(energy, nai, label='nai')
# for material_index in range(len(materials)):
#     plt.plot(i_want_list[list(m_chi).index(10.)][0][0], i_want_list[list(m_chi).index(10.)][0][1][material_index],
#              label=f'pdf_{materials[material_index]}')
# plt.legend()
# plt.show()
#
# plt.yscale('log')
# # plt.xlim(0, 40)
# # plt.ylim(10**(-12), 10**0)
# plt.plot(energy, na, label='na')
# plt.plot(energy, iod, label='iod')
# plt.plot(energy, nai, label='nai')
# plt.plot(i_want_list[list(m_chi).index(10.)][3][0], i_want_list[list(m_chi).index(10.)][3][1], label='pdf_sum')
# plt.legend()
# plt.show()
#
# plt.yscale("log")
# # plt.xlim(0, 40)
# # plt.ylim(10**(-12), 10**0)
# plt.plot(energy, na, label='na')
# plt.plot(energy, iod, label='iod')
# plt.plot(energy, nai, label='nai')
# plt.plot(i_want_list[list(m_chi).index(10.)][6][0], i_want_list[list(m_chi).index(10.)][6][1], label='pdf_sum_convoluted')
# plt.legend()
# plt.show()
#
stop = timeit.default_timer()
print('Time: ', stop - start)

# # print('thr = 0.0301 and res = 0.0046:')  # Optional
# # print('min DM mass in GeV = ', omega._find_dm_min(materials))
# # TODO: ist aber dieser dm_min nicht für WIMP Modell? Vielleict mass range einfach dem User überlassen ->
# #  andere Modells, und dann diese Funktion als zusätlicher Tool lassen, falls man die min Masse braucht im WIMP modell
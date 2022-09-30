import numpy as np
import timeit
from matplotlib import pyplot as plt
from Ilok import Ilok
from Ilok_2 import Ilok2

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

m_chi = np.geomspace(1., 100., 50)
m_chi = list(m_chi)
for mass in x_axis_plot:
    m_chi.append(mass)
m_chi.append(10.)
m_chi.sort()
m_chi = list(dict.fromkeys(m_chi))
m_chi = np.array(m_chi)
# m_chi = [10.]
# m_chi = m_chi[(m_chi > 14) & (m_chi < 90)]
sample_size = 10000
materials = [[40.078, 40.078/287.914, 'Ca', 'C3P1_DetA_eff_AR_Ca.dat'],
             [183.84, 183.84/287.914, 'W', 'C3P1_DetA_eff_AR_W.dat'],
             [15.999, 15.999*4/287.914, 'O', 'C3P1_DetA_eff_AR_O.dat']]
# materials = [[23., 23/150, 'Na', 'C3P1_DetA_eff_AR_Ca.dat'],
#              [127., 127/150, 'I', 'C3P1_DetA_eff_AR_Ca.dat']]

# omega = Ilok2()
omega = Ilok()
omega.set_detector(5.594, 0.0046, 0.0301, materials, 16.)
omega.set_cut_eff('C3P1_DetA_cuteff.dat')
omega.set_masses(m_chi)
omega.set_sigma_interval(1e-6, 1e-2)
omega.set_mu_interval(0.25, 250)
omega.set_confidence_level(0.9)
omega.get_data('C3P1_DetA_AR.dat')  # TODO: data schon in AR, braucht man dann die Multiplikation von AR??
omega.set_table_variables(False, 100, 'table5')  # 'table' mit mu bis 250 und N = 1000, 'table2' mit mu bis 100 und N = 100, table3 mu linspaced 100 n 1000
omega.get_limit()
stop = timeit.default_timer()
print('Time: ', stop - start)

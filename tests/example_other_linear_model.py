"""
Code example number 2.
Use this template if:
i) you are using another model for calculating the pdf, cdf and the expected number of events (μ).
ii) there is a linear dependence between the expected number of events (μ) and the cross-sections (σ).
"""

import numpy as np
import pyyellin as yell
import timeit
from scipy.stats import norm

######### Mock cdf and mus #########
# This part is not really relevant for usage. Here only to be able to run this code with mock data.
m_chi = np.geomspace(1., 20., 20)
sigmas = np.geomspace(1e-6, 1e-2, 1000)
mus = [5000*x for x in sigmas]  # wichtig dass die hinzuzufügenden mus bis zumindest omega.set_mu_interval max gehen!
mus_1 = [(0.005-50)/(1e-2-1e-6)*x+50-1e-6*(0.005-50)/(1e-2-1e-6) for x in sigmas]
energies = np.arange(0.0005, 30.0005, 0.0005)
cdf = norm.cdf(energies, 2, 30.001/30)
cdf_list = [[energies, np.array(norm.cdf(energies, 4+np.sqrt(mass), 1))] for mass in m_chi]
####################################

start = timeit.default_timer()

############### Main ###############
# The main part, which you will change and use.
m_chi = np.geomspace(1., 20., 20)  # Define masses.
omega = yell.Ilok()  # Create instance
omega.add_sigmas_and_mus(sigmas, mus)  # Add cross-sections (σ) and corresponding expected number of events (μ).
omega.add_cdf(cdf_list)  # Add list [[energies, cdf1], [energies, cdf2],...]
omega.set_masses(m_chi)  # Set dark matter masses.
omega.set_mu_interval(0.25, 50)  # Set the min and max values for μ. Max(μ) should be around Max(mus).
omega.set_confidence_level(0.9)  # Set confidence level.
omega.get_data('C3P1_DetA_AR.dat')  # Set the path for data.
omega.set_table_variables(True, 100, 'table')  # Set whether or not you want to create a new table, how many lists there should be per mu and the path for tabulated data.
omega.get_limit_from_other_linear_model()  # Call the function to determine limit values.
print(vars(omega).keys())  # Callable variables.
####################################

stop = timeit.default_timer()
print('Time: ', stop - start)

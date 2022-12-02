"""
Code example number 1.
Use this template if you are using our model for calculating the pdf, cdf and the expected number of events (μ).
"""

import numpy as np
import pyyellin as yell
import timeit

start = timeit.default_timer()

############### Main ###############
m_chi = np.geomspace(1., 100., 25)  # Define masses.
sample_size = 10000  # Define sample size for rvs functions.
materials = [[40.078, 40.078/287.914, 'Ca'], [183.84, 183.84/287.914, 'W'], [15.999, 15.999*4/287.914, 'O']]  # Define the materials of the detector.
omega = yell.Ilok()  # Create instance
omega.set_detector(5.689, 0.0046, 0.0301, materials, 16., 0.0005)  # Set detector parameters exposure, resolution, threshold, materials, upper integral limit and energy grid step size.
omega.set_cut_eff('C3P1_DetA_cuteff.dat')  # Set the path for the cut efficiency data.
omega.set_masses(m_chi)  # Set dark matter masses.
omega.set_sampling_size(sample_size)  # Set sample size. Optional, only needed for rvs functions.
omega.set_sigma_interval(1e-6, 1e-2)  # Set the min and max values for cross-sections (σ).
omega.set_mu_interval(0.25, 50)  # Set the min and max values for μ. Max(μ) should be around Max(mus).
# item_list = omega.calculate_items()  # Optional. Declare the results you want to be returned. Parameters (boolean): pdf, cdf, samples, pdf_sum, cdf_sum, samples_sum, pdf_sum_convoluted, cdf_sum_convoluted, samples_sum_convoluted, mus.
omega.set_confidence_level(0.9)  # Set confidence level.
omega.get_data('C3P1_DetA_AR.dat')  # Set the path for data.
omega.set_table_variables(False, 100, 'table')  # Set whether or not you want to create a new table, how many lists there should be per mu and the path for tabulated data.
omega.get_limit_optimum_interval()  # Call the function to determine limit values.
print(vars(omega).keys())  # Callable variables.
####################################

stop = timeit.default_timer()
print('Time: ', stop - start)


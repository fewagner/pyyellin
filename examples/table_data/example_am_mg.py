import os
import numpy as np
import pyyellin as yell
from pathlib import Path
from scipy.stats import norm


def main_example_am_mg_linear():
    """
    todo !!!!!!!!!!!!!!
    :return:
    """
    ######### Mock cdf and mus #########
    # This part is not really relevant for usage. Here only to be able to run this code with mock data.
    m_chi = np.geomspace(1., 100., 5)
    sigmas = np.geomspace(1e-6, 1e-2, 1000)
    mus = [[5000*x/mass for x in sigmas] for mass in m_chi]  # wichtig dass die hinzuzufügenden mus bis zumindest omega.set_mu_interval max gehen!
    mus_1 = [[(0.005-50)/(1e-2-1e-6)*x/mass+50-1e-6*(0.005-50)/(1e-2-1e-6) for x in sigmas] for mass in m_chi]
    energies = np.arange(0.0005, 30.0005, 0.0005)
    cdf = norm.cdf(energies, 2, 30.001/30)
    cdf_list = [[energies, np.array(norm.cdf(energies, 4+np.sqrt(mass), 1))] for mass in m_chi]
    ####################################

    m_chi = np.geomspace(1., 100., 5)  # Define masses.
    omega = yell.ModeLimit()  # Create instance
    omega.add_sigmas_and_mus(sigmas, mus_1)  # Add cross-sections (σ) and corresponding expected number of events (μ).
    omega.add_cdf(cdf_list)  # Add list [[energies, cdf1], [energies, cdf2],...]
    omega.set_masses(m_chi)  # Set dark matter masses.
    omega.set_mu_interval(0.25, 50, 1000)  # Set the min and max values for μ. Max(μ) should be around Max(mus).
    omega.set_confidence_level(0.9)  # Set confidence level.
    omega.get_data(Path(os.getcwd() + '\example_data\cresst_III\C3P1_DetA_AR.dat'))  # Set the path for data.
    omega.set_table_and_results_paths(Path(os.getcwd() + '/table_data/'), Path(os.getcwd() + '/results/'))  # Set directory paths for tables and results.
    omega.set_table_variables(False, 100, 'table')  # Set whether or not you want to create a new table, how many lists there should be per mu and the path for tabulated data.
    omega.get_limit_from_another_model_maximum_gap()  # Call the function to determine limit values.
    return


def main_example_am_mg_nonlinear():
    """
    todo !!!!!!!!!!!!!
    :return:
    """
    ######### Mock cdf and mus #########
    # This part is not really relevant for usage. Here only to be able to run this code with mock data.
    m_chi = np.geomspace(1., 100., 5)
    sigmas = np.geomspace(1e-6, 1e-2, 1000)
    mus = [[-np.sqrt(x)*np.log(x)*20*5.5/mass for x in sigmas] for mass in m_chi]  # wichtig dass die hinzuzufügenden mus bis zumindest omega.set_mu_interval max gehen!
    mus_1 = [[-x*x*np.sin(x)*np.log(x)*1e8/9/mass for x in sigmas] for mass in m_chi]
    energies = np.arange(0.0005, 30.0005, 0.0005)
    cdf = norm.cdf(energies, 2, 30.001/30)
    cdf_list = [[energies, np.array(norm.cdf(energies, 4+np.sqrt(mass), 1+mass/7))] for mass in m_chi]
    ####################################

    m_chi = np.geomspace(1., 100., 5)  # Define masses.
    omega = yell.ModeLimit()  # Create instance
    omega.add_sigmas_and_mus(sigmas, mus_1)  # Add cross-sections (σ) and corresponding expected number of events (μ).
    omega.add_cdf(cdf_list)  # Add list [[energies, cdf1], [energies, cdf2],...]
    omega.set_masses(m_chi)  # Set dark matter masses.
    omega.set_mu_interval(0.25, 50, 1000)  # Set the min and max values for μ. Max(μ) should be around Max(mus).
    omega.set_confidence_level(0.9)  # Set confidence level.
    omega.get_data(Path(os.getcwd() + '\example_data\cresst_III\C3P1_DetA_AR.dat'))  # Set the path for data.
    omega.set_table_and_results_paths(Path(os.getcwd() + '/table_data/'), Path(os.getcwd() + '/results/'))  # Set directory paths for tables and results.
    omega.set_table_variables(False, 100, 'table')  # Set whether or not you want to create a new table, how many lists there should be per mu and the path for tabulated data.
    omega.get_limit_from_another_model_maximum_gap()  # Call the function to determine limit values.
    return


if __name__ == '__main__':
    main_example_am_mg_linear()
    main_example_am_mg_nonlinear()

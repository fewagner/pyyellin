import os
import numpy as np
import pyyellin as yell
from pathlib import Path


def main_example_sm_oi_1():
    """
    Code example for limit calculation using cdfs modelled using integrated signal model, hence sm, and optimum interval
    method, hence oi.
    Use this template if you are using our model for calculating the pdf, cdf and the expected number of events (μ).
    This example uses the CRESST-III data.
    :return:
    """
    m_chi = np.geomspace(1., 100., 5)  # Define masses.
    materials = [[40.078, 40.078/287.914, 'Ca', Path(os.getcwd() + '\example_data\cresst_III\C3P1_DetA_eff_AR_Ca.dat')],
                 [183.84, 183.84/287.914, 'W', Path(os.getcwd() + '\example_data\cresst_III\C3P1_DetA_eff_AR_W.dat')],
                 [15.999, 15.999*4/287.914, 'O', Path(os.getcwd() + '\example_data\cresst_III\C3P1_DetA_eff_AR_O.dat')]]  # Define the materials of the detector.
    omega = yell.ModeLimit()  # Create instance
    omega.set_detector(5.689, 0.0046, 0.0301, materials, 16., 0.0005)  # Set detector parameters exposure, resolution, threshold, materials, upper integral limit and energy grid step size.
    omega.set_cut_eff(Path(os.getcwd() + '\example_data\cresst_III\C3P1_DetA_cuteff.dat'))  # Set the path for the cut efficiency data.
    omega.set_masses(m_chi)  # Set dark matter masses.
    omega.set_sigma_interval(1e-6, 1e-2)  # Set the min and max values for cross-sections (σ).
    omega.set_mu_interval(0.25, 50, 1000)  # Set the min and max values for μ. Max(μ) should be around Max(mus).
    omega.set_confidence_level(0.9)  # Set confidence level.
    omega.get_data(Path(os.getcwd() + '\example_data\cresst_III\C3P1_DetA_AR.dat'))  # Set the path for data.
    omega.set_table_and_results_paths(Path(os.getcwd() + '/table_data/'), Path(os.getcwd() + '/results/'))  # Set directory paths for tables and results.
    omega.set_table_variables(False, 100, 'table')  # Set whether or not you want to create a new table, how many lists there should be per mu and the path for tabulated data.
    omega.get_limit_optimum_interval()  # Call the function to determine limit values using the optimum interval method.
    return


def main_example_sm_oi_2():
    """
    Code example for limit calculation using cdfs modelled using integrated signal model, hence sm, and optimum interval
    method, hence oi.
    Use this template if you are using our model for calculating the pdf, cdf and the expected number of events (μ).
    This example uses the CRESST-II Lise data.
    :return:
    """
    m_chi = np.geomspace(1., 100., 5)  # Define masses.
    materials = [[40.078, 40.078/287.914, 'Ca', Path(os.getcwd() + '\example_data\cresst_II_lise\Lise_eff_AR_Ca.dat')],
                 [183.84, 183.84/287.914, 'W', Path(os.getcwd() + '\example_data\cresst_II_lise\Lise_eff_AR_W.dat')],
                 [15.999, 15.999*4/287.914, 'O', Path(os.getcwd() + '\example_data\cresst_II_lise\Lise_eff_AR_O.dat')]]  # Define the materials of the detector.
    omega = yell.ModeLimit()  # Create instance
    omega.set_detector(52.15, 0.062, 0.307, materials, 40., 0.0005)  # Set detector parameters exposure, resolution, threshold, materials, upper integral limit and energy grid step size.
    omega.set_cut_eff(1)  # Set the path for the cut efficiency data. In this case the cut efficiency is accounted for in material efficiency data, so we set the parameter equal to 1.
    omega.set_masses(m_chi)  # Set dark matter masses.
    omega.set_sigma_interval(1e-6, 1e-2)  # Set the min and max values for cross-sections (σ).
    omega.set_mu_interval(0.25, 50, 1000)  # Set the min and max values for μ. Max(μ) should be around Max(mus).
    omega.set_confidence_level(0.9)  # Set confidence level.
    omega.get_data(Path(os.getcwd() + '\example_data\cresst_II_lise\Lise_AR.dat'))  # Set the path for data.
    omega.set_table_and_results_paths(Path(os.getcwd() + '/table_data/'), Path(os.getcwd() + '/results/'))  # Set directory paths for tables and results.
    omega.set_table_variables(False, 100, 'table')  # Set whether or not you want to create a new table, how many lists there should be per mu and the path for tabulated data.
    omega.get_limit_optimum_interval()  # Call the function to determine limit values using the optimum interval method.
    return


if __name__ == '__main__':
    main_example_sm_oi_1()
    main_example_sm_oi_2()

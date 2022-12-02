import os
import numpy as np
import pyyellin as yell
from pathlib import Path


def main_example_sm_1():
    """
    Code example for signal modeling, hence sm.
    Use this template if you want to calculate and return pdf, cdf, rvs samples using cdf, pdf_sum, cdf_sum, rvs samples
    using cdf_sum and mus.
    This example uses the CRESST-III data.
    :return:
    """
    m_chi = np.geomspace(1., 100., 5)  # Define masses.
    sample_size = 10000  # Define sample size for rvs functions.
    materials = [[40.078, 40.078/287.914, 'Ca', Path(os.getcwd() + '\example_data\cresst_III\C3P1_DetA_eff_AR_Ca.dat')],
                 [183.84, 183.84/287.914, 'W', Path(os.getcwd() + '\example_data\cresst_III\C3P1_DetA_eff_AR_W.dat')],
                 [15.999, 15.999*4/287.914, 'O', Path(os.getcwd() + '\example_data\cresst_III\C3P1_DetA_eff_AR_O.dat')]]  # Define the materials of the detector.
    omega = yell.ModeLimit()  # Create instance
    omega.set_detector(5.689, 0.0046, 0.0301, materials, 16., 0.0005)  # Set detector parameters exposure, resolution, threshold, materials, upper integral limit and energy grid step size.
    omega.set_cut_eff(Path(os.getcwd() + '\example_data\cresst_III\C3P1_DetA_cuteff.dat'))  # Set the path for the cut efficiency data.
    omega.set_masses(m_chi)  # Set dark matter masses.
    omega.set_sampling_size(sample_size)  # Set sample size. Optional, only needed for rvs functions.
    omega.set_sigma_interval(1e-6, 1e-2)  # Set the min and max values for cross-sections (σ).
    omega.set_mu_interval(0.25, 50, 1000)  # Set the min and max values for μ. Max(μ) should be around Max(mus).
    item_list = omega.calculate_items(pdf_bool=True, cdf_bool=True, samples_bool=True, pdf_sum_bool=True, cdf_sum_bool=True, samples_sum_bool=True, mus_bool=True)  # Declare the results you want to be returned. Parameters (boolean): pdf, cdf, samples, pdf_sum, cdf_sum, samples_sum, mus. Default values are True for all.
    return item_list


def main_example_sm_2():
    """
    Code example for signal modeling, hence sm.
    Use this template if you want to calculate and return pdf, cdf, rvs samples using cdf, pdf_sum, cdf_sum, rvs samples
    using cdf_sum and mus.
    This example uses the CRESST-II Lise data.
    :return:
    """
    m_chi = np.geomspace(1., 100., 5)  # Define masses.
    sample_size = 10000  # Define sample size for rvs functions.
    materials = [[40.078, 40.078/287.914, 'Ca', Path(os.getcwd() + '\example_data\cresst_II_lise\Lise_eff_AR_Ca.dat')],
                 [183.84, 183.84/287.914, 'W', Path(os.getcwd() + '\example_data\cresst_II_lise\Lise_eff_AR_W.dat')],
                 [15.999, 15.999*4/287.914, 'O', Path(os.getcwd() + '\example_data\cresst_II_lise\Lise_eff_AR_O.dat')]]  # Define the materials of the detector.
    omega = yell.ModeLimit()  # Create instance
    omega.set_detector(52.15, 0.062, 0.307, materials, 40., 0.0005)  # Set detector parameters exposure, resolution, threshold, materials, upper integral limit and energy grid step size.
    omega.set_cut_eff(1)  # Set the path for the cut efficiency data. In this case the cut efficiency is accounted for in material efficiency data, so we set the parameter equal to 1.
    omega.set_masses(m_chi)  # Set dark matter masses.
    omega.set_sampling_size(sample_size)  # Set sample size. Optional, only needed for rvs functions.
    omega.set_sigma_interval(1e-6, 1e-2)  # Set the min and max values for cross-sections (σ).
    omega.set_mu_interval(0.25, 50, 1000)  # Set the min and max values for μ. Max(μ) should be around Max(mus).
    item_list = omega.calculate_items(pdf_bool=True, cdf_bool=True, samples_bool=True, pdf_sum_bool=True, cdf_sum_bool=True, samples_sum_bool=True, mus_bool=True)  # Declare the results you want to be returned. Parameters (boolean): pdf, cdf, samples, pdf_sum, cdf_sum, samples_sum, mus. Default values are True for all.
    return item_list


if __name__ == '__main__':
    main_example_sm_1()
    main_example_sm_2()

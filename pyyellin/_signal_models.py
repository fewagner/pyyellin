
import numpy as np
from scipy.stats import expon

class SignalModel:

    def __init__(self):
        # Here global parameters for the signal model can be fixed. E.g. the escape velocity, ...
        pass

    def set_detector(self, exposure: float, resolution: float, threshold: float):

        self.exposure = exposure
        self.resolution = resolution
        self.threshold = threshold

    def pdf(self, x, pars):
        """
        The probability density function of the signal model, evaluated at a given grid.

        :param x: The grid for the evaluation.
        :type x: list
        :param pars: The signal parameters.
        :type pars: list
        :return: The evaluated probability density function on the grid.
        :rtype: list
        """
        raise NotImplemented('You need to implement the pdf function!')

    def cdf(self, x, pars):
        """
        The cummulative density function of the signal model, evaluated at a given grid.

        :param x: The grid for the evaluation.
        :type x: list
        :param pars: The signal parameters.
        :type pars: list
        :return: The evaluated cummulative density function on the grid.
        :rtype: list
        """
        raise NotImplemented('You need to implement the cdf function!')

    def rvs(self, size, pars):
        """
        Draw a sample from the signal model.

        :param size: The sample size
        :type size: int
        :param pars: The signal parameters.
        :type pars: list
        :return: The recoil energies (and possible other parameters) of the drawn sample.
        :rtype: list
        """
        raise NotImplemented('You need to implement the rvs function!')

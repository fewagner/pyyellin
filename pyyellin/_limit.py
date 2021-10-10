import numpy as np


class Limit:
    """
    A class to calculate Limits with Yellins Optimum Interval Method.
    """

    def __init__(self):

        self.data = []
        self.efficiencies = []
        self.eff_grids = []
        self.acc_regions = []
        self.signal_models = []
        self.resolutions = []
        self.thresholds = []
        self.exposures = []

    # ------------------------------------------------
    # User API
    # ------------------------------------------------

    def hello(self):
        # TODO delete this function
        print('Hello World!')

    def add_data(self, data: list, efficiency: list, eff_grid: list, acc_region: tuple, signal_model: object,
                 resolution: float, threshold: float, exposure: float):
        """
        Add data and definitions of a new experiment to the limit calculation.

        :param data: The recoil energies of the measured event, in keV.
        :type data: list
        :param efficiency: The binned and sorted survival probabilities of the detector.
        :type efficiency: list
        :param eff_grid: The energies corresponding to the survival probabilities, in keV.
        :type eff_grid: list
        :param acc_region: The lower and upper energy limit of the detector
        :type acc_region: tuple
        :param signal_model: A model of the expected dark matter signal for this detector.
        :type signal_model: list
        :param resolution: The resolution of the detector, in keV.
        :type resolution: list
        :param threshold: The lower threshold of the detector, in keV.
        :type threshold: list
        :param exposure: The exposure of the measurement, in kg days.
        :type exposure: list
        """

        # TODO do some asserts if the data formats are alright
        # assert statement_that_must_be_true, 'This error message is shown if the statement is not True.'

        # TODO do some assert if the Signal model is a child of the class SignalModel
        # TODO check efficiencies list is sorted
        # TODO check if eff_grid is sorted
        # TODO ...

        self.data.append(np.array(data))
        self.efficiencies.append(np.array(efficiency))
        self.eff_grid.append(np.array(eff_grid))
        self.acc_regions.append(np.array(acc_region))
        self.signal_models.append(signal_model)
        self.resolutions.append(resolution)
        self.thresholds.append(threshold)
        self.exposures.append(exposure)

        assert 1>2, "test"


    def get_limit(self, signal_pars: list):
        """
        Here the limit is calculated and returned.

        :param signal_pars: List of the signal parameters for all signal models for the individual experiments.
        :type signal_pars: list
        :return: The limit for the given signal parameters.
        :rtype: float
        """

        assert len(signal_pars) == len(self.data)

        # TODO call here the function for the limit calculation
        pass


    # ------------------------------------------------
    # private
    # ------------------------------------------------

    # here you can put function that are not callable for the user
    # these always start with an underscore:
    # def _dummy_private_func():
    #   ...
from typing import Any, List,Tuple
from function_regression.functions.base_function import BaseFunction
import numpy as np
import exputils as eu
from numpy.polynomial import Polynomial

class SinusodialFunction(BaseFunction):

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()

        # defines how long a certain action is kept
        def_config.in_features = 2
        def_config.ranges = (-5,5)
        def_config.peak_distr_ranges = (-2,2) ## should be integer to avoid leakage
        def_config.difficulty = 1 ### describes the frequency of the function. Should be an integer to avoid leakage          
        def_config.sample_rates = [10,10]

        return def_config


    def __init__(self, config=None, **kwargs):
   
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        if isinstance(self.config.ranges, tuple):
            self.config.ranges = [self.config.ranges] * self.config.in_features

        if isinstance(self.config.peak_distr_ranges, tuple):
            self.config.peak_distr_ranges = [self.config.peak_distr_ranges] * self.config.in_features

        assert self.config.coef.shape[1] == self.config.in_features, "in_features must be the same size as dim of the means"

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        input = args[0]

        value = 0

        freq = self.config.difficulty
        window = self.config.peak_distr_ranges

        in_window = True

        for dim in range(len(input)):
            if input[dim] > window[dim][0] and input[dim] < window[dim][1]:
                value = value + np.sin(2*np.pi*freq*input[dim])
            else:
                in_window = False
            ### we only use one dimension for the sinusodial function
            break

        if not in_window:
            value = 0


        return value



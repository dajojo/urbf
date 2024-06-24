from typing import Any, List,Tuple
from function_regression.functions.base_function import BaseFunction
import numpy as np
import exputils as eu

class MichalewiczFunction(BaseFunction):

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()

        # defines how long a certain action is kept
        def_config.in_features = 2
        def_config.ranges = (-5,5)
        def_config.peak_distr_ranges = (-5,5)
        def_config.difficulty = 2
        def_config.coef = np.array([[5,15],[5,15]])
        
        def_config.sample_rates = [5,5]

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
        #input = np.array(input)
        value = 0

        coef = self.config.coef.transpose()

        for dim in range(len(coef)):
            value = value - (np.sin(input[dim]) * np.sin((1*input[dim]**2)/np.pi)**(2*10)) 

        return value



from typing import Any, List,Tuple
from function_regression.functions.base_function import BaseFunction
import numpy as np
import exputils as eu
from numpy.polynomial import Polynomial



class DiscontinuousFunction(BaseFunction):

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()

        def_config.in_features = 2
        def_config.ranges = (-5,5)
        def_config.peak_distr_ranges = (-5,5)
        def_config.difficulty = 2
        def_config.coef = np.array([[-1,1],[0,0]])
        
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
        input = np.array(args[0])

        value = 0

        means = np.array(self.config.means)
        height = np.array(self.config.coef)
        depth = np.array(self.config.stds)
        peak_distr_ranges = np.array(self.config.peak_distr_ranges)

        for step in range(len(means)):
            mean = means[step]
            range_scale = np.array([(peak_distr_ranges[dim][1] - peak_distr_ranges[dim][0])/2 for dim in range(input.shape[0])])
            scaled_depth = depth[step] * range_scale

            # Calculate the Euclidean distance from the mean
            distance = np.linalg.norm(input - mean)

            # Check if the distance is within the scaled depth
            if distance < np.linalg.norm(scaled_depth):
                value += height[step][0]

        return value



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
        seeds = np.array(self.config.stds)  # Used as seeds
        peak_distr_ranges = np.array(self.config.peak_distr_ranges)

        for step, seed in enumerate(seeds):
            np.random.seed(int(seed*1000))  # Set the seed for reproducibility

            mean = means[step]
            range_scale = np.array([(peak_distr_ranges[dim][1] - peak_distr_ranges[dim][0]) / 2 for dim in range(input.shape[0])])

            # Generate random radii and rotation angle
            radius = np.random.rand(input.shape[0]) * range_scale
            rotation_angle = np.random.rand() * 2 * np.pi  # Rotation angle between 0 and 2π

            # Create rotation matrix for 2D (extend this for higher dimensions if needed)
            cos_angle, sin_angle = np.cos(rotation_angle), np.sin(rotation_angle)
            rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

            # Rotate the input and mean
            rotated_input = rotation_matrix.dot(input - mean)
            rotated_radius = rotation_matrix.dot(radius)

            # Adjusted distance calculation for a rotated ellipse
            scaled_input = rotated_input / rotated_radius
            distance_squared = np.sum(scaled_input ** 2)

            # Check if the point lies within the rotated ellipse
            if distance_squared < 1:
                height = np.random.rand()  # You can also randomize height if needed
                value += height

        return value



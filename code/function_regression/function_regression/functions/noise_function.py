from typing import Any, List,Tuple
from function_regression.functions.base_function import BaseFunction
import numpy as np
import exputils as eu

from typing import Any, List, Tuple
import numpy as np
import exputils as eu

class WhiteNoiseFunction(BaseFunction):

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()
        def_config.in_features = 2  # Update this for n-dimensional support
        def_config.ranges = [(-5, 5)] 
        def_config.peak_distr_ranges = [(0, 1)] 
        def_config.difficulty = 2
        def_config.coef = np.array([[5]] )
        def_config.sample_rates = [5]
        return def_config

    def generate_nd_white_noise(self, dimensions, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(size=dimensions) * 100

    def apply_frequency_limit(self, noise, sample_rates, cutoff_frequency_hz, dimensions):

        
        assert all([noise.shape[0] == noise.shape[i] for i in range(len(noise.shape))]), "Noise shape is not equal in all dimensions"
        assert len(sample_rates) == len(dimensions), "Sample rates and dimensions are not equal"
        assert all([sample_rates[0] == sample_rate for sample_rate in sample_rates]), "Sample rates are not equal"
        

        cutoff_frequency_hz = cutoff_frequency_hz * (noise.shape[0]/sample_rates[0]**2) ##### Sample rate and shape has to be the same

        # Fourier transform to frequency domain
        noise_fft = np.fft.fftn(noise)
        noise_fft_shifted = np.fft.fftshift(noise_fft)  # Shift the zero frequency component to the center

        # Calculate center of the frequency domain
        center = tuple(dim // 2 for dim in dimensions)

        # Create an n-dimensional sphere mask
        mask = np.zeros(dimensions, dtype=bool)
        
        for indices in np.ndindex(*dimensions):
            distance = sum(((center[dim] - indices[dim]) / sample_rates[dim])**2 for dim in range(len(dimensions)))
            mask[indices] = distance <= cutoff_frequency_hz**2

        print(mask)

        # Apply the mask (filter)
        noise_fft_shifted *= mask

        # Shift back and inverse Fourier transform to spatial domain
        noise_fft_original = np.fft.ifftshift(noise_fft_shifted)
        filtered_noise = np.fft.ifftn(noise_fft_original).real

        return filtered_noise


    def __init__(self, config=None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        if isinstance(self.config.ranges, tuple):
            self.config.ranges = [self.config.ranges] * self.config.in_features

        if isinstance(self.config.peak_distr_ranges, tuple):
            self.config.peak_distr_ranges = [self.config.peak_distr_ranges] * self.config.in_features

        if len(self.config.sample_rates) < self.config.in_features:
            self.config.sample_rates = [self.config.sample_rates[0]] * self.config.in_features

        assert self.config.coef.shape[1] == self.config.in_features, "in_features must be the same size as dim of the means"
        

        # Adjust for n dimensions
        dimensions = [(self.config.ranges[i][1] - self.config.ranges[i][0]) * self.config.sample_rates[i] for i in range(self.config.in_features)]
        self.noise = self.generate_nd_white_noise(dimensions, abs(int(1000 * self.config.coef[0][0])))
        self.noise = self.apply_frequency_limit(self.noise, self.config.sample_rates, self.config.difficulty, dimensions)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        input = args[0]

        input_point = np.array(input)
        in_range = ([self.config.peak_distr_ranges[i][0] <= input_point[i] <= self.config.peak_distr_ranges[i][1] for i in range(self.config.in_features)])

        if not all(in_range):
            return 0
        
        indices = [int(input[i] * self.config.sample_rates[i]) for i in range(self.config.in_features)]

        return self.noise[tuple(indices)]




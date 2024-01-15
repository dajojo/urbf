from typing import Any, List,Tuple
from function_regression.functions.base_function import BaseFunction
import numpy as np
import exputils as eu

class WhiteNoiseFunction(BaseFunction):

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

    def generate_2d_white_noise(self,width, height, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(size=(width, height)) * 100


    def apply_frequency_limit(self,noise, sample_rate, cutoff_frequency_hz, width, height):
        
        cutoff_frequency_hz = cutoff_frequency_hz* (width/sample_rate) /( sample_rate ) #* width/sample_rate

        # Fourier transform to frequency domain
        noise_fft = np.fft.fft2(noise)
        noise_fft_shifted = np.fft.fftshift(noise_fft)  # Shift the zero frequency component to the center

        # Apply low-pass filter
        # u, v = np.meshgrid(range(-width//2, width//2), range(-height//2, height//2))
        # d = np.sqrt(u**2 + v**2)
        # mask = d <= cutoff_frequency_hz
        # noise_fft_shifted *= mask

        # Create a radial mask for the low-pass filter
        u, v = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        d = np.sqrt(u**2 + v**2)
        mask = d <= cutoff_frequency_hz

        # Apply the mask (filter)
        noise_fft_shifted *= mask

        # Shift back and inverse Fourier transform to spatial domain
        noise_fft_original = np.fft.ifftshift(noise_fft_shifted)
        filtered_noise = np.fft.ifft2(noise_fft_original).real

        return filtered_noise


    def __init__(self, config=None, **kwargs):
   
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        if isinstance(self.config.ranges, tuple):
            self.config.ranges = [self.config.ranges] * self.config.in_features

        if isinstance(self.config.peak_distr_ranges, tuple):
            self.config.peak_distr_ranges = [self.config.peak_distr_ranges] * self.config.in_features

        assert self.config.coef.shape[1] == self.config.in_features, "in_features must be the same size as dim of the means"
        
        width = (self.config.ranges[0][1] - self.config.ranges[0][0])*self.config.sample_rates[0]
        height = (self.config.ranges[1][1] - self.config.ranges[1][0])*self.config.sample_rates[1]

        self.noise = self.generate_2d_white_noise(width,height , int(1000*self.config.coef[0][0]).__abs__() )
        self.noise = self.apply_frequency_limit(self.noise, self.config.sample_rates[0], self.config.difficulty, width,height )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        input = args[0]

        x = input[0]
        y = input[1]

        value = self.noise[int(x*self.config.sample_rates[0]),int(y*self.config.sample_rates[1])]

        return value



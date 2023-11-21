from typing import Any, List,Tuple
from function_regression.functions.base_function import BaseFunction
import numpy as np
import exputils as eu

class GaussianMixtureFunction(BaseFunction):

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()

        # defines how long a certain action is kept
        def_config.in_features = 2
        def_config.ranges = (-5,5)
        
        def_config.difficulty = 2
        def_config.means = np.array([[1,1],[-1,-1]])
        def_config.vars = np.array([1,1])
        def_config.coef = np.array([1,1])
        
        def_config.sample_rates = [10,10]

        return def_config


    #def __init__(self,in_features: int ,means, vars,coef = None,):
    def __init__(self, config=None, **kwargs):
   
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        if isinstance(self.config.ranges, tuple):
            self.config.ranges = [self.config.ranges] * self.config.in_features

        print(self.config.means)
        print(self.config.vars)

        assert len(self.config.means) == len(self.config.vars), "Number of means must match the number of vars"
        assert len(self.config.means) > 0 and len(self.config.vars) > 0, "At least one gaussian has to be given"
        assert self.config.means.shape[1] == self.config.in_features, "in_features must be the same size as dim of the means"

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        input = args[0]

        value = 0

        for i in range(len(self.config.means)):
            value = value + np.exp(-0.5 * (np.linalg.norm(input - self.config.means[i]) / self.config.vars[i]) ** 2)

        return value



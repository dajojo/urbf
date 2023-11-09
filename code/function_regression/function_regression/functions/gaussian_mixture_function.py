from typing import Any, List,Tuple
from function_regression.functions.base_function import BaseFunction
import numpy as np

class GaussianMixtureFunction(BaseFunction):
    def __init__(self,in_features: int ,means, vars,coef = None,):
        
        self.in_features = in_features

        self.means = means 
        self.vars = vars

        if coef == None:
            self.coef = [1]*len(self.means)
        else:
            self.coef = coef            

        assert len(self.means) == len(self.vars), "Number of means must match the number of vars"
        assert len(self.means) > 0 and len(self.vars) > 0, "At least one gaussian has to be given"
        assert self.means.shape[1] == self.in_features, "in_features must be the same size as dim of the means"

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        input = args[0]

        value = 0

        for i in range(len(self.means)):
            value = value + np.exp(-0.5 * (np.linalg.norm(input - self.means[i]) / self.vars[i]) ** 2)

        return value



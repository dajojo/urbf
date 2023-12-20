from typing import Any, List,Tuple
import matplotlib.pyplot as plt
import torch
import exputils as eu
import numpy as np
from sklearn.cross_decomposition import PLSRegression

class PLS:
    @staticmethod
    def default_config():
        def_config = eu.AttrDict()

        def_config.in_features = 2
        def_config.out_features = 1
        #def_config.hidden_features = [16,16,8,4]
        def_config.ranges = [(-5,5),(-5,5)]
        def_config.sample_rates = [100,100]
        #def_config.use_urbf = True

        return def_config


    #def __init__(self, in_features:int, out_features:int, hidden_features:List[int]):
    def __init__(self, config=None, **kwargs):
        super().__init__()
        self.config = eu.combine_dicts(kwargs, config, self.default_config())
        self.regr = PLSRegression(n_components=self.config.in_features)

    def fit(self,x,y):
        print(f"PLS: x -> {x.shape} y -> {y.shape}")
        self.regr = self.regr.fit(x.detach().numpy(), y.detach().numpy())


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return torch.tensor(self.regr.predict(*args)).unsqueeze(-1)
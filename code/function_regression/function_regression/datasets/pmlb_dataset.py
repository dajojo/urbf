from typing import Any, List,Tuple
import numpy as np
import exputils as eu
from pmlb import fetch_data

class PMLBDataset():

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()
        def_config.name = "1028_SWD"

        return def_config

    def __init__(self, config=None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())


    def generate_samples(self) -> Tuple:
        X,Y = fetch_data(self.config.name, return_X_y=True,)
        Y = np.expand_dims(Y, axis=1)

        print(f"Sampled X:{X.shape} Y:{Y.shape} from {self.config.name}")

        return X,Y
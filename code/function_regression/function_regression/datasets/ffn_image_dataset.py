from typing import Any, List,Tuple
import numpy as np
import exputils as eu
import plotly.graph_objects as go
import os, imageio
import torch

class FFNImageDataset():

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()
        def_config.name = "div2k" ## or 'text'
        def_config.in_features = 2
        def_config.out_features = 3
        def_config.max_samples = -1
        return def_config

    def __init__(self, config=None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())


    def load_dataset(self,filename, id):

        filename = "../../../" + filename


        if not os.path.exists(filename):
            os.system(f"gdown --id {id}")

        npz_data = np.load(filename)
        out = {
            "data_grid_search":npz_data['train_data'] / 255.,
            "data_test":npz_data['test_data'] / 255.,
        }
        return out

    def generate_samples(self) -> Tuple:

        if self.config.name == 'div2k':
            datasets = self.load_dataset('data_div2k.npz', '1TtwlEDArhOMoH18aUyjIMSZ3WODFmUab')
        elif self.config.name == 'text':
            datasets = self.load_dataset('data_2d_text.npz', '1V-RQJcMuk9GD4JCUn70o7nwQE0hEzHoT')

        RES = 512
        y_train = datasets['data_grid_search'][0]
        x1 = np.linspace(0, 1, RES+1)[:-1]
        x_train = np.stack(np.meshgrid(x1,x1), axis=-1)

        x_train = np.reshape(x_train,(-1,2))
        y_train = np.reshape(y_train,(-1,3))

        return x_train, y_train

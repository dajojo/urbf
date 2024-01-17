from typing import Any, List,Tuple
import matplotlib.pyplot as plt
import torch
import exputils as eu
import numpy as np
import torch.nn as nn
from urbf_layer.adaptive_urbf_layer import AdaptiveURBFLayer

from urbf_layer.urbf_layer import URBFLayer

class FFNMLP(torch.nn.Module):


    @staticmethod
    def default_config():
        def_config = eu.AttrDict()

        def_config.in_features = 2
        def_config.out_features = 1
        def_config.hidden_features = [16,16,8,4]
        def_config.range = (-5,5)
        def_config.use_sigmoid = False
        def_config.scale = 10
        def_config.univariate = False
        return def_config

    def __init__(self, config=None, **kwargs):
        super().__init__()

        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        if isinstance(self.config.range, tuple):
            self.config.range = [self.config.range] * self.config.in_features

        self.layers = []
   
        mapping_size = int(self.config.hidden_features[0] // 2)

        self.layers.append(FFNLayer(in_features=self.config.in_features,mapping_size=mapping_size,scale=self.config.scale,univariate=self.config.univariate))

        if self.config.scale == None:
            self.layers.append(torch.nn.Linear(in_features=self.config.in_features,out_features=self.config.hidden_features[0]))


        self.layers.append(torch.nn.ReLU())

        in_dim = self.config.hidden_features[0]
        for hidden_dim in self.config.hidden_features[1:]:
            self.layers.append(torch.nn.Linear(in_dim, hidden_dim))
            self.layers.append(torch.nn.ReLU())
            in_dim = hidden_dim
    
        self.layers.append(torch.nn.Linear(in_dim, self.config.out_features))
        

        if self.config.use_sigmoid:
            self.layers.append(torch.nn.Sigmoid()) ##### This is as per the original code from the paper but it behaves poorly for the discontinuous functions


        self.layers = torch.nn.Sequential(*self.layers)

        self.params = nn.ModuleDict({
             'mlp': nn.ModuleList(self.layers),
            })


    def forward(self,x):
        return self.layers(x)





class FFNLayer(torch.nn.Module):
        
    def __init__(self,in_features,mapping_size,scale=None,univariate=False) -> None:
        super().__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size
        self.B = None

        if scale != None:
            B_gauss = np.random.normal(size=(mapping_size, in_features))
            self.B = torch.from_numpy(B_gauss * scale).float()

            if univariate:
                ### additionally use a uniform distribution to decide which dimension to use
                self.univariate_mapping = torch.zeros((mapping_size,in_features))
                for i in range(mapping_size):
                    self.univariate_mapping[i,np.random.randint(0,in_features)] = 1.0

                self.B = self.B * self.univariate_mapping


    def forward(self,x):
        return self.input_mapping(x)

    def input_mapping(self,x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        if self.B is not None:
            x_proj = ((2. * (np.pi) * x) @ self.B.T)
            if isinstance(x_proj, np.ndarray):
                x_proj = torch.from_numpy(x_proj)

            x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).float()
    
        return x
from typing import Any, List,Tuple
import matplotlib.pyplot as plt
import torch
import exputils as eu
import numpy as np
import torch.nn as nn
from urbf_layer.adaptive_urbf_layer import AdaptiveURBFLayer

from urbf_layer.rbf_layer import RBFLayer

class RBFMLP(torch.nn.Module):

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()

        def_config.in_features = 2
        def_config.out_features = 1
        def_config.hidden_features = [16,16,8,4]
        def_config.range = (-5,5)
        def_config.use_rbf = True
        def_config.univariate = True

        return def_config


    def __init__(self, config=None, **kwargs):
        super().__init__()

        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        if isinstance(self.config.range, tuple):
            self.config.range = [self.config.range] * self.config.in_features

        self.layers = []

        if self.config.use_rbf:
            self.layers.append(RBFLayer(in_features=self.config.in_features,out_features=self.config.hidden_features[0],data_range=self.config.range,univariate=self.config.univariate))
        else:
            self.layers.append(torch.nn.Linear(in_features=self.config.in_features,out_features=self.config.hidden_features[0]))
            self.layers.append(torch.nn.ReLU())


        in_dim = self.config.hidden_features[0]
        for hidden_dim in self.config.hidden_features[1:]:
            self.layers.append(torch.nn.Linear(in_dim, hidden_dim))
            self.layers.append(torch.nn.ReLU())
            in_dim = hidden_dim
    
        self.layers.append(torch.nn.Linear(in_dim, self.config.out_features))

        self.layers = torch.nn.Sequential(*self.layers)

        self.params = nn.ModuleDict({
             'rbf': nn.ModuleList([self.layers[0]]) if self.config.use_rbf else nn.ModuleList([]),
             'rbf_linear': nn.ModuleList([*((self.layers[0].linear_layer,) if hasattr(self.layers[0],"linear_layer") else ())]),
             'mlp': nn.ModuleList([*self.layers[1:]]) if self.config.use_rbf else nn.ModuleList(self.layers),
            })


    def forward(self,x):
        return self.layers(x)





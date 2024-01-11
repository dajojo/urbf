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
        def_config.ranges = (-5,5)
        def_config.sample_rates = 100 ### Only for plotting purposes... can be removed later
        def_config.use_sigmoid = False
        def_config.scale = 10
        return def_config

    def __init__(self, config=None, **kwargs):
        super().__init__()

        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        if isinstance(self.config.ranges, tuple):
            self.config.ranges = [self.config.ranges] * self.config.in_features

        if isinstance(self.config.sample_rates, int):
            self.config.sample_rates = [self.config.sample_rates] * self.config.in_features
     
        self.layers = []

   
        first_hidden_features = self.config.hidden_features[0]

        #reduced_first_layer_size = int((2 + N_u)*N_u / (N_in + N_u)) #### TODO: Adjust the first layer size to match the num of parameters in the URBFLayer

        self.layers.append(FFNLayer(in_features=self.config.in_features,mapping_size=first_hidden_features,scale=self.config.scale))

        if self.config.scale != None:
            self.layers.append(torch.nn.Linear(in_features=self.config.in_features * first_hidden_features,out_features=first_hidden_features))
        else:
            self.layers.append(torch.nn.Linear(in_features=self.config.in_features,out_features=first_hidden_features))


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



    ### For plotting purposes only...
    # def generate_samples(self) -> Tuple:

    #     assert len(self.config.ranges) == self.config.in_features, "Sample Range elements must match the number of in_features"
    #     assert len(self.config.sample_rates) == self.config.in_features, "Sample Rate elements must match the number of in_features"

    #     # Generating a meshgrid for multi-dimensional sampling
    #     axes = [np.arange(r[0], r[1], 1/s) for r, s in zip(self.config.ranges, self.config.sample_rates)]
    #     meshgrid = np.meshgrid(*axes, indexing='ij')
    #     flat_grid = np.stack([axis.flat for axis in meshgrid], axis=-1)
        
    #     # Compute the values using vectorized operations
    #     values = np.array([self.forward(torch.from_numpy(point).to(torch.float32)).detach().numpy() for point in flat_grid])


    #     return np.transpose(np.array(meshgrid),(1,2,0)), values.reshape([*meshgrid[0].shape,1])

    # def plot(self):

    #     points,values = self.generate_samples()

    #     assert len(points.shape) - 1 <= 3, "Can only plot functions for dim <= 3" 

    #     fig = plt.figure()
    #     ax = plt.axes(projection='3d')

    #     ax.contour3D(points[...,0], points[...,1], values[...,0], 50, cmap='binary')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     ax.set_zlabel('z')


class FFNLayer(torch.nn.Module):
        
    def __init__(self,in_features,mapping_size,scale=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size
        self.B = None

        if scale != None:
            B_gauss = np.random.normal(size=(mapping_size, in_features))
            self.B = torch.from_numpy(B_gauss * scale).float()
            print(f"B: {self.B.shape}")
        else:
            print("no scale provided")

    def forward(self,x):
        return self.input_mapping(x)

    def input_mapping(self,x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        if self.B is None:
            in_channels = self.in_features
        else:
            x_proj = ((2. * (np.pi) * x) @ self.B.T)
            if isinstance(x_proj, np.ndarray):
                x_proj = torch.from_numpy(x_proj)

            x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).float()
        
            in_channels = self.in_features * self.mapping_size


        return x
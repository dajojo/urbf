from typing import Any, List,Tuple
import matplotlib.pyplot as plt
import torch
import exputils as eu
import numpy as np

from urbf_layer.urbf_layer import URBFLayer

class URBFMLP(torch.nn.Module):


    @staticmethod
    def default_config():
        def_config = eu.AttrDict()

        def_config.in_features = 2
        def_config.out_features = 1
        def_config.hidden_features = [16,16,8,4]
        def_config.ranges = [(-5,5),(-5,5)]
        def_config.sample_rates = [100,100]
        def_config.use_urbf = True

        return def_config


    #def __init__(self, in_features:int, out_features:int, hidden_features:List[int]):
    def __init__(self, config=None, **kwargs):
        super().__init__()

        self.config = eu.combine_dicts(kwargs, config, self.default_config())

        self.layers = []


        if self.config.use_urbf:
            self.layers.append(URBFLayer(in_features=self.config.in_features,out_features=self.config.hidden_features[0]))
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


    def forward(self,x):
        return self.layers(x)



    def generate_samples(self) -> Tuple:

        assert len(self.config.ranges) == self.config.in_features, "Sample Range elements must match the number of in_features"
        assert len(self.config.sample_rates) == self.config.in_features, "Sample Rate elements must match the number of in_features"

        # Generating a meshgrid for multi-dimensional sampling
        axes = [np.arange(r[0], r[1], 1/s) for r, s in zip(self.config.ranges, self.config.sample_rates)]
        meshgrid = np.meshgrid(*axes, indexing='ij')
        flat_grid = np.stack([axis.flat for axis in meshgrid], axis=-1)
        
        # Compute the values using vectorized operations
        values = np.array([self.forward(torch.from_numpy(point).to(torch.float32)).detach().numpy() for point in flat_grid])


        return np.transpose(np.array(meshgrid),(1,2,0)), values.reshape([*meshgrid[0].shape,1])

    def plot(self):

        points,values = self.generate_samples()

        assert len(points.shape) - 1 <= 3, "Can only plot functions for dim <= 3" 

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.contour3D(points[...,0], points[...,1], values[...,0], 50, cmap='binary')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')



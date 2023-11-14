from typing import Any, List,Tuple
import matplotlib.pyplot as plt
import numpy as np


class BaseFunction():
    def __init__(self):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def generate_samples(self) -> Tuple:

        assert len(self.config.ranges) == self.config.in_features, "Sample Range elements must match the number of in_features"
        assert len(self.config.sample_rates) == self.config.in_features, "Sample Rate elements must match the number of in_features"

        # Generating a meshgrid for multi-dimensional sampling
        axes = [np.arange(r[0], r[1], 1/s) for r, s in zip(self.config.ranges, self.config.sample_rates)]
        meshgrid = np.meshgrid(*axes, indexing='ij')
        flat_grid = np.stack([axis.flat for axis in meshgrid], axis=-1)
        
        # Compute the values using vectorized operations
        values = np.array([self.__call__(point) for point in flat_grid])


        return np.transpose(np.array(meshgrid),(1,2,0)).reshape((-1,len(self.config.ranges))), values.reshape([*meshgrid[0].shape,1]).reshape((-1,1))
    
    def plot(self):

        points,values = self.generate_samples()

        assert len(points.shape) - 1 <= 3, "Can only plot functions for dim <= 3" 

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        ax.contour3D(points[...,0], points[...,1], values[...,0], 50, cmap='binary')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        #ax.view_init(60, 35)
        
from typing import Any, List,Tuple
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go


class BaseFunction():
    def __init__(self):
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError

    def generate_samples(self, ranges=None) -> Tuple:
        if ranges is None:
            ranges = self.config.ranges

        num_dimensions = self.config.in_features

        assert num_dimensions == self.config.in_features, "Sample Range elements must match the number of in_features"

        # Generating a meshgrid for multi-dimensional sampling
        axes = [np.arange(r[0], r[1], 1/s) for r, s in zip(ranges, self.config.sample_rates)]
        meshgrid = np.meshgrid(*axes, indexing='ij')
        flat_grid = np.stack([axis.flat for axis in meshgrid], axis=-1)
        
        # Compute the values using vectorized operations
        values = np.array([self.__call__(point) for point in flat_grid])

        # Reshape the output
        grid_shape = meshgrid[0].shape
        points = np.transpose(np.array(meshgrid), tuple(range(1, num_dimensions + 1)) + (0,))
        points = points.reshape((-1, num_dimensions))
        values = values.reshape([*grid_shape, 1]).reshape((-1, 1))

        return points, values


    # def generate_samples(self,ranges=None) -> Tuple:

    #     if ranges == None:
    #         ranges = self.config.ranges

    #     assert len(ranges) == self.config.in_features, "Sample Range elements must match the number of in_features"
    #     assert len(self.config.sample_rates) == self.config.in_features, "Sample Rate elements must match the number of in_features"

    #     # Generating a meshgrid for multi-dimensional sampling
    #     axes = [np.arange(r[0], r[1], 1/s) for r, s in zip(ranges, self.config.sample_rates)]
    #     meshgrid = np.meshgrid(*axes, indexing='ij')
    #     flat_grid = np.stack([axis.flat for axis in meshgrid], axis=-1)
        
    #     # Compute the values using vectorized operations
    #     values = np.array([self.__call__(point) for point in flat_grid])


    #     return np.transpose(np.array(meshgrid),(1,2,0)).reshape((-1,len(ranges))), values.reshape([*meshgrid[0].shape,1]).reshape((-1,1))
    

    def plot(self):
        points, values = self.generate_samples()
        print(points.shape)

        num_dimensions = points.shape[-1]#len(points.shape) - 1

        assert num_dimensions <= 3, "Can only plot functions for dim <= 3"

        if num_dimensions == 1:
            fig = go.Figure(data=[go.Scatter(
                x=points[:, 0],
                y=values[:, 0],
                mode='markers',
                marker=dict(
                    size=2,
                    color=values[:, 0],  # Coloring based on the values
                    colorscale='Viridis',  # Color scale
                    opacity=0.8
                )
            )])
            fig.update_layout(title="1D Scatter Plot")
        elif num_dimensions == 2:
            fig = go.Figure(data=[go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=values[:, 0],
                mode='markers',
                marker=dict(
                    size=2,
                    color=values[:, 0],  # Coloring based on the values
                    colorscale='Viridis',  # Color scale
                    opacity=0.8
                )
            )])
            fig.update_layout(scene=dict(
                xaxis_title='X',
                yaxis_title='Y'
            ))
            fig.update_layout(title="2D Scatter Plot")
        else:
            fig = go.Figure(data=[go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=values[:, 0],  # Coloring based on the values
                    colorscale='Viridis',  # Color scale
                    opacity=0.8
                )
            )])
            fig.update_layout(scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ))
            fig.update_layout(title="3D Scatter Plot")

        fig.show()

    def plot_matplt(self,title="",showcolorbar=True):
        points, values = self.generate_samples()
        print(points.shape)

        num_dimensions = points.shape[-1]

        assert num_dimensions <= 3, "Can only plot functions for dim <= 3"

        assert num_dimensions <= 3, "Can only plot functions for dim <= 3"

        if num_dimensions == 1:
            plt.scatter(points[:, 0], values[:, 0], c=values[:, 0], cmap='viridis', s=2)
            plt.colorbar(label='Value')
            plt.title("1D Scatter Plot")
            plt.xlabel("X")
            plt.ylabel("Value")
            plt.show()

        elif num_dimensions == 2:
            # Create grid for surface plot
            x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
            y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
            
            # Interpolate values for the grid
            from scipy.interpolate import griddata
            zz = griddata(points, values[:, 0], (xx, yy), method='cubic')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(xx, yy, zz, cmap='viridis', edgecolor='none')
            
            if showcolorbar:
                fig.colorbar(surf, ax=ax, label='Value')
        
            ax.set_title(title)
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("y")
            plt.show()

        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            img = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=values[:, 0], cmap='viridis', s=2)
            fig.colorbar(img, ax=ax, label='Value')
            ax.set_title("3D Scatter Plot")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.show()
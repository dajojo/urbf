from typing import Any, List,Tuple
import numpy as np
import exputils as eu
from pmlb import fetch_data
import plotly.graph_objects as go

class PMLBDataset():

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()
        def_config.name = "1028_SWD"
        def_config.in_features = 10
        return def_config

    def __init__(self, config=None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())


    def generate_samples(self) -> Tuple:
        print(f"fetching: {self.config.name}")
        X,Y = fetch_data(self.config.name, return_X_y=True,)
        Y = np.expand_dims(Y, axis=1)

        print(f"Sampled X:{X.shape} Y:{Y.shape} from {self.config.name}")

        return X,Y
    
    def plot(self):

        points,values = self.generate_samples()

        assert len(points.shape) - 1 <= 3, "Can only plot functions for dim <= 3" 


        print(points.shape)
        print(values.shape)

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

        fig.show()
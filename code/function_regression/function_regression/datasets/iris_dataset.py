from typing import Any, List,Tuple
import numpy as np
import exputils as eu
import plotly.graph_objects as go
from sklearn import datasets

class IrisDataset():

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()
        #def_config.name = "1028_SWD"
        def_config.in_features = 4
        def_config.max_samples = 10000
        return def_config

    def __init__(self, config=None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())


    def generate_samples(self) -> Tuple:
        print(f"fetching: iris dataset")
        iris = datasets.load_iris()
        
        X = iris.data
        Y = iris.target.T

        max_samples = self.config.max_samples


        X = X[:np.min([X.shape[0],max_samples])]
        Y = Y[:np.min([Y.shape[0],max_samples])]

        Y = np.expand_dims(Y, axis=1)

        print(f"Sampled X:{X.shape} Y:{Y.shape} from iris dataset")

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
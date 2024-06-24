from typing import Any, List,Tuple
import numpy as np
import exputils as eu
import plotly.graph_objects as go
from sklearn.datasets import fetch_california_housing





class HousingDataset():

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()
        #def_config.name = "1028_SWD"
        def_config.in_features = 8
        def_config.max_samples = 10000
        return def_config

    def __init__(self, config=None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())


    def generate_samples(self) -> Tuple:
        print(f"fetching: housing dataset")
        housing = fetch_california_housing(data_home="../../../data/cal_housing")#X,Y = fetch_data(self.config.name, return_X_y=True,) ## -> X: (n_samples, n_features), Y: (n_samples,)

        X = housing.data
        Y = housing.target

        print(f"housing dataset: {X.shape} {Y.shape}")

        max_samples = self.config.max_samples


        X = X[:np.min([X.shape[0],max_samples])]
        Y = Y[:np.min([Y.shape[0],max_samples])]

        Y = np.expand_dims(Y, axis=1)

        print(f"Sampled X:{X.shape} Y:{Y.shape} from {self.config.name}")


        ## normalize the data
        X = (X - np.min(X,axis=0)) / (np.max(X,axis=0) - np.min(X,axis=0))
        Y = (Y - np.min(Y,axis=0)) / (np.max(Y,axis=0) - np.min(Y,axis=0))

        #X = X[:,-2:]

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
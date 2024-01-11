from typing import Any, List,Tuple
import numpy as np
import exputils as eu
import plotly.graph_objects as go
import os, imageio

class ImageDataset():

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()
        def_config.name = "-"
        def_config.in_features = 2
        def_config.max_samples = -1
        return def_config

    def __init__(self, config=None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())

    def generate_samples(self) -> Tuple:
        # Download image, take a square crop from the center
        image_url = 'https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg'
        img = imageio.imread(image_url)[..., :3] / 255.
        c = [img.shape[0]//2, img.shape[1]//2]
        r = 128
        img = img[c[0]-r:c[0]+r, c[1]-r:c[1]+r]

        # Create input pixel coordinates in the unit square
        coords = np.linspace(0, 1, img.shape[0], endpoint=False)
        x_test = np.stack(np.meshgrid(coords, coords), -1)
        test_data = [x_test, img]

        return test_data[0].reshape(-1,2), test_data[1].reshape(-1,3)
    
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
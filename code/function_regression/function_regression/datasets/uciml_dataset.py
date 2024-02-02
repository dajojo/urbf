from typing import Any, List,Tuple
import numpy as np
import exputils as eu
from ucimlrepo import fetch_ucirepo 
import plotly.graph_objects as go
import pandas as pd

uciml_dataset_names = ['Auto MPG', 
                       'Breast Cancer Wisconsin (Original)', 
                       'Glass Identification', 
                       'Heart Disease', 
                       'Hepatitis', 
                       'Liver Disorders', 
                       'Lung Cancer', 
                       'Optical Recognition of Handwritten Digits', 
                       'Spambase', 
                       'Wine', 
                       'Zoo', 
                       'Wine Quality', 
                       'Parkinsons Telemonitoring', 
                       'Diabetic Retinopathy Debrecen', 
                       'Heart failure clinical records', 
                       'Myocardial infarction complications', 
                       'Sepsis Survival Minimal Clinical Records', 
                       'AIDS Clinical Trials Group Study 175', 
                       'CDC Diabetes Health Indicators', 
                       'National Poll on Healthy Aging (NPHA)']


# [
#     'Abalone', 'Adult', 'Auto MPG', 'Automobile', 'Breast Cancer', 
#     'Breast Cancer Wisconsin (Original)', 'Breast Cancer Wisconsin (Diagnostic)', 
#     'Car Evaluation', 'Credit Approval', 'Computer Hardware', 'Covertype', 
#     'Glass Identification', 'Heart Disease', 'Hepatitis', 'Image Segmentation', 
#     'Ionosphere', 'Iris', 'Letter Recognition', 'Liver Disorders', 'Lung Cancer', 
#     'Mushroom', 'Optical Recognition of Handwritten Digits', 'Spambase', 
#     'Congressional Voting Records', 'Wine', 'Yeast', 'Zoo', 
#     'Statlog (German Credit Data)', 'MAGIC Gamma Telescope', 'Wine Quality', 
#     'Parkinsons Telemonitoring', 'Bank Marketing', 'ILPD (Indian Liver Patient Dataset)', 
#     'Bike Sharing Dataset', 'Thoracic Surgery Data', 'Diabetes 130-US hospitals for years 1999-2008', 
#     'Diabetic Retinopathy Debrecen', 'Heart failure clinical records', 
#     'Estimation of obesity levels based on eating habits and physical condition', 
#     'Rice (Cammeo and Osmancik)', 'Bone marrow transplant: children', 'HCV data', 
#     'Myocardial infarction complications', 'Dry Bean Dataset', 
#     "Predict students' dropout and academic success", 'Glioma Grading Clinical and Mutation Features', 
#     'Sepsis Survival Minimal Clinical Records',
#     'SUPPORT2', 'National Health and Nutrition Health Survey 2013-2014 (NHANES) Age Prediction Subset', 
#     'AIDS Clinical Trials Group Study 175', 'CDC Diabetes Health Indicators', 
#     'Infrared Thermography Temperature', 'National Poll on Healthy Aging (NPHA)', 
#     'Regensburg Pediatric Appendicitis'
# ]


class UCIMLDataset():

    @staticmethod
    def default_config():
        def_config = eu.AttrDict()
        def_config.name = "1028_SWD"
        def_config.in_features = 10
        def_config.max_samples = 10000
        return def_config

    def __init__(self, config=None, **kwargs):
        self.config = eu.combine_dicts(kwargs, config, self.default_config())


    def generate_samples(self) -> Tuple:
        print(f"fetching: {self.config.name}")
        dataset = fetch_ucirepo(name=self.config.name) ## -> X: (n_samples, n_features), Y: (n_samples,)

        features_df = dataset.data.features
        target_df = dataset.data.targets


        # Assuming 'features_df' is your features DataFrame and 'target_df' is your target DataFrame

        # Temporarily concatenate the DataFrames along the columns
        combined_df = pd.concat([features_df, target_df], axis=1)

        # Drop rows with NaN values
        combined_df_clean = combined_df.dropna()

        # Separate the DataFrames again
        features_df_clean = combined_df_clean[features_df.columns]
        target_df_clean = combined_df_clean[target_df.columns]

        X = features_df_clean.to_numpy().astype(np.float32)
        Y = target_df_clean.to_numpy().astype(np.float32)

        max_samples = self.config.max_samples

        X = X[:np.min([X.shape[0],max_samples])]
        Y = Y[:np.min([Y.shape[0],max_samples])]

        #Y = np.expand_dims(Y, axis=1)

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
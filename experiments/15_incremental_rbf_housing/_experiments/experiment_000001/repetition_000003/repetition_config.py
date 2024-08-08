import numpy as np
import exputils as eu
import function_regression

config = eu.AttrDict(
    # random seed for the repetition
    seed = 1234 + 3,


    model = eu.AttrDict(
        cls=function_regression.models.SVR,
        in_features=8,
        out_features=1,
        use_rbf=False,
        complexity=None,
        hidden_features=[],
        range=(None,None),
        scale=0,
        univariate=False,
        ),

    dataset = eu.AttrDict(
        cls=function_regression.datasets.HousingDataset,
        ),
    
    trainer = eu.AttrDict(
        cls=function_regression.SKLFit,
        learning_rate=0.01,
        rbf_learning_rate=0,
        n_epochs=1000,
        batch_size=128),

)

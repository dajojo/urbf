import numpy as np
import exputils as eu
import function_regression

config = eu.AttrDict(
    # random seed for the repetition
    seed = 1234 + 7,


    model = eu.AttrDict(
        cls=function_regression.models.FFNMLP,
        in_features=8,
        out_features=1,
        use_rbf=False,
        complexity=None,
        hidden_features=[64,32,16,8],
        range=(None,None),
        scale=1,
        univariate=True,
        ),

    dataset = eu.AttrDict(
        cls=function_regression.datasets.HousingDataset,
        ),
    
    trainer = eu.AttrDict(
        cls=function_regression.AdamTrainer,
        learning_rate=0.0001,
        rbf_learning_rate=0,
        n_epochs=1000,
        batch_size=128),

)

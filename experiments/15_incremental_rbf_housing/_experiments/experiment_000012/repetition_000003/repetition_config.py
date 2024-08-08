import numpy as np
import exputils as eu
import function_regression

config = eu.AttrDict(
    # random seed for the repetition
    seed = 1234 + 3,


    model = eu.AttrDict(
        cls=function_regression.models.IncrementalRBFMLP,
        in_features=8,
        out_features=1,
        use_rbf=True,
        complexity='diagonal',
        hidden_features=[64,32,16,8],
        range=(None,None),
        scale=0,
        univariate=False,
        ),

    dataset = eu.AttrDict(
        cls=function_regression.datasets.HousingDataset,
        ),
    
    trainer = eu.AttrDict(
        cls=function_regression.AdamTrainer,
        learning_rate=0.0001,
        rbf_learning_rate=0.0001,
        n_epochs=1000,
        batch_size=128),

)

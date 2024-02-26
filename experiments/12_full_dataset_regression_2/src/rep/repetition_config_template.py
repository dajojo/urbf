import numpy as np
import exputils as eu
import function_regression

config = eu.AttrDict(
    # random seed for the repetition
    seed = 1234 + <repetition_id>,


    model = eu.AttrDict(
        cls=function_regression.models.<model>,
        in_features=<in_features>,
        use_rbf=<use_rbf>,
        univariate=<univariate>,
        hidden_features=<hidden_features>,
        range=<model_range>,
        scale=<scale>
        ),

    dataset = eu.AttrDict(
        cls=function_regression.datasets.<dataset>,
        name=<name>
        ),
    
    trainer = eu.AttrDict(
        cls=function_regression.<trainer>,
        learning_rate=<learning_rate>,
        rbf_learning_rate=<rbf_learning_rate>,
        n_epochs=<n_epochs>,
        batch_size=<batch_size>),

)

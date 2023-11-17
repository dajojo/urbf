import numpy as np
import exputils as eu
import function_regression

config = eu.AttrDict(
    # random seed for the repetition
    seed = 1234 + <repetition_id>,

    model = eu.AttrDict(
        cls=function_regression.models.<model>,
        hidden_features=<hidden_features>,
        use_urbf=<use_urbf>,
        in_features=<in_features>,
        ),

    dataset = eu.AttrDict(
        cls=function_regression.datasets.<dataset>,
        name=<name>
        ),
    
    trainer = eu.AttrDict(
        cls=function_regression.<trainer>,
        learning_rate=<learning_rate>,
        n_epochs=<n_epochs>,
        batch_size=<batch_size>),
)

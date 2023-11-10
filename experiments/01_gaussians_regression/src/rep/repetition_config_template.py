import numpy as np
import exputils as eu
import function_regression

config = eu.AttrDict(
    # random seed for the repetition
    seed = 1234 + <repetition_id>,

    model = eu.AttrDict(
        cls=function_regression.models.<model>,
        hidden_features=<hidden_features>,
        use_urbf=<use_urbf>
        ),

    function = eu.AttrDict(
        cls=function_regression.functions.<function>,
        difficulty=<difficulty>,
        in_features=<in_features>
        ),
    trainer = eu.AttrDict(
        cls=function_regression.Trainer,
        learning_rate=<learning_rate>,
        n_epochs=<n_epochs>,
        batch_size=<batch_size>),
)

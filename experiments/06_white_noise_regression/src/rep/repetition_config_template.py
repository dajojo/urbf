import numpy as np
import exputils as eu
import function_regression

config = eu.AttrDict(
    # random seed for the repetition
    seed = 1234 + <repetition_id>,


    model = eu.AttrDict(
        cls=function_regression.models.<model>,
        in_features=<in_features>,
        out_features=<out_features>,
        use_rbf=<use_rbf>,
        univariate=<univariate>,
        difficulty=<difficulty>,
        hidden_features=<hidden_features>,
        range=<model_range>,
        scale=<scale>,
        initial_distribution=<initial_distribution>,
        learnable=<learnable>,
        ),

    function = eu.AttrDict(
        cls=function_regression.functions.<function>,
        difficulty=<difficulty>,
        in_features=<in_features>,
        ranges=<func_ranges>,
        peak_distr_ranges=<peak_distr_ranges>,
        sample_rates = [20]
        ),
    
    trainer = eu.AttrDict(
        cls=function_regression.<trainer>,
        learning_rate=<learning_rate>,
        rbf_learning_rate=<rbf_learning_rate>,
        n_epochs=<n_epochs>,
        batch_size=<batch_size>),

)

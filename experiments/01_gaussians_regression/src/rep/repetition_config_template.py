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
        use_split_merge=<use_split_merge>,
        split_merge_temperature=<split_merge_temperature>,
        in_features=<in_features>,
        ranges=<model_ranges>,
        init_with_spektral=<init_with_spektral>,
        use_adaptive_range=<use_adaptive_range>,
        ),

    function = eu.AttrDict(
        cls=function_regression.functions.<function>,
        difficulty=<difficulty>,
        in_features=<in_features>,
        ranges=<func_ranges>,
        peak_distr_ranges=<peak_distr_ranges>
        ),
    
    trainer = eu.AttrDict(
        cls=function_regression.<trainer>,
        learning_rate=<learning_rate>,
        urbf_learning_rate = <urbf_learning_rate>,
        n_epochs=<n_epochs>,
        batch_size=<batch_size>),
)

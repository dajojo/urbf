import numpy as np
import exputils as eu
import rl_lib


config = eu.AttrDict(
    # random seed for the repetition
    seed = 3487 + <repetition_id>,

    # agent configuration
    agent = eu.AttrDict(
        cls = rl_lib.agent.<agent>,
        n_steps_keep_action = <n_steps_keep_action>,
        ),

    # environment configuration
    env = eu.AttrDict(
        cls = rl_lib.env.Corridor,
        length = 9,
        left_reward = 0,
        right_reward = 1,
        movement_punishment = -0.1
    ),

    # number of episodes per experiment
    n_episodes = <n_episodes>,

    # maximum number of steps per episode
    n_max_steps_per_epsiode = 100,
)

from function_regression.function_sample_dataset import FunctionSampleDataset
from function_regression.datasets.pmlb_dataset import PMLBDataset
from function_regression.models.urbf_mlp import URBFMLP
from function_regression.sgd_trainer import SGDTrainer
import numpy as np
from typing import Any, List,Tuple
from sklearn.model_selection import train_test_split
import exputils as eu
import exputils.data.logging as log


def sample_random_arrays(n: int, ranges: List[Tuple[float, float]]) -> np.ndarray:
    """
    Sample n random D-dimensional arrays.

    :param n: Number of arrays to sample.
    :param ranges: List of tuples, where each tuple contains the min and max values of the range for a dimension.
    :return: A (n, D) shaped array of random samples.
    """
    # Check that ranges is a list of tuples
    if not all(isinstance(r, tuple) and len(r) == 2 for r in ranges):
        raise ValueError("Ranges must be a list of (min, max) tuples.")
    
    # Number of dimensions is determined by the length of the ranges list
    D = len(ranges)
    
    # Sample random values within the specified ranges for each dimension
    samples = np.array([np.random.uniform(r[0], r[1], n) for r in ranges])
    
    # Transpose the samples to get the desired shape (n, D)
    return samples.T#.reshape(-1,D)



def run_data_experiment(config=None, **kwargs):
    # define the default configuration
    default_config = eu.AttrDict(
        model = eu.AttrDict(cls=URBFMLP),
        dataset = eu.AttrDict(
            cls=PMLBDataset,
            name="1028_SWD"),
        trainer = eu.AttrDict(cls=SGDTrainer),
        seed = 123,
        test_split_size = 0.2,
        val_split_size = 0.2,
        log_to_tensorboard = True,
    )

    # set the config based on the default config, given config, and the given function arguments
    config = eu.combine_dicts(kwargs, config, default_config)

    # set random seeds with seed defined in the config
    eu.misc.seed(config)

    ##
    dataset = eu.misc.create_object_from_config(config.dataset)
    trainer = eu.misc.create_object_from_config(config.trainer)
    model = eu.misc.create_object_from_config(config.model)
    
    sample_points, sample_values = dataset.generate_samples()
    print(f"Sampled {sample_values.shape}")
    
    #function.plot()   

    # Split the dataset into training (60%), validation (20%), and test (20%)
    train_points, test_points, train_values, test_values = train_test_split(
        sample_points, sample_values, test_size=0.2, random_state=config.seed)

    # Further split the training set into training and validation sets
    train_points, val_points, train_values, val_values = train_test_split(
        train_points, train_values, test_size= config.val_split_size / (1 - config.test_split_size) , random_state=config.seed)

    train_dataset = FunctionSampleDataset(train_points, train_values)
    test_dataset = FunctionSampleDataset(test_points, test_values)

    model = trainer.train(model,train_dataset,test_dataset)
    
    return model


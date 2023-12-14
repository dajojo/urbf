from function_regression.function_sample_dataset import FunctionSampleDataset
from function_regression.datasets.pmlb_dataset import PMLBDataset
from function_regression.models.urbf_mlp import URBFMLP
from function_regression.sgd_trainer import SGDTrainer
import numpy as np
from typing import Any, List,Tuple
from sklearn.model_selection import train_test_split
import exputils as eu
import exputils.data.logging as log
import time
from pmlb import classification_dataset_names, regression_dataset_names
import os

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



def run_data_experiments(config=None, **kwargs):
    print("iterate over a list of datasets and replace this field by their name")
    ### implement checkpoints
    ### implement logging for different datasets

    datasets = regression_dataset_names

    log_config = eu.AttrDict(
        directory = eu.DEFAULT_DATA_DIRECTORY,
    )
    summary_logger = log.Logger(log_config)

    for dataset_name in datasets:

        ### if the dataset is already trained, skip it... check if the folder is present
        working_dir = eu.DEFAULT_DATA_DIRECTORY+dataset_name
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        else:
            print(f"Dataset {dataset_name} already trained, skipping...")
            continue

        log_config = eu.AttrDict(
            directory = working_dir,
        )
        logger = log.Logger(log_config)


        config.dataset.name = dataset_name

        dataset = eu.misc.create_object_from_config(config.dataset)
        sample_points, sample_values = dataset.generate_samples()

        print(sample_points.shape)

        min = np.min(sample_points,axis=0)
        max = np.max(sample_points,axis=0)


        config.model.in_features = sample_points.shape[-1]
        config.model.ranges = list(zip(min,max))

        print(config.trainer)

        model = run_data_experiment(config=config,logger=logger, **kwargs)

        summary_logger.add_object("dataset",dataset_name)

        test_loss = logger.numpy_data["test_loss"]
        min_test_loss = np.min(test_loss)

        summary_logger.add_value("min_test_loss",min_test_loss)
        summary_logger.add_value("duration",logger.numpy_data["duration"][-1])
        summary_logger.save()


        


def run_data_experiment(config=None,logger=None, **kwargs):

    if logger == None:
        logger = log
    

    start_time = time.time()

    # define the default configuration
    default_config = eu.AttrDict(
        model = eu.AttrDict(cls=URBFMLP,            
            in_features=2,
            use_urbf=False,
            ranges=(-10,5),   
            use_split_merge=False,
            split_merge_temperature=0.1,     
            use_back_tray=False,
            back_tray_ratio = 0.5),
        dataset = eu.AttrDict(
            cls=PMLBDataset,
            name="feynman_III_12_43"),
        trainer = eu.AttrDict(cls=SGDTrainer,        
            learning_rate=0.1,
            urbf_learning_rate=1,
            n_epochs=1000,
            batch_size=64),
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

    print(config.model)
    
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

    model = trainer.train(model,train_dataset,test_dataset,logger=logger)
    
    end_time = time.time()

    duration = end_time - start_time

    print(f"Duration: {time.strftime('%H:%M:%S', time.gmtime(duration))}s")
    logger.add_value('duration', duration)
    logger.save()

    return model


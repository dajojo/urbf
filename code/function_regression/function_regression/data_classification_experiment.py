from math import sqrt
from function_regression.datasets.uciml_dataset import UCIMLDataset, uciml_dataset_names
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
import torch

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



def run_data_classification_experiments(config=None, **kwargs):
    print("iterate over a list of datasets and replace this field by their name")
    ### implement checkpoints
    ### implement logging for different datasets

    print(config.dataset.cls)

    if config.dataset.cls is UCIMLDataset:
        datasets = uciml_dataset_names
    elif config.dataset.cls is PMLBDataset:
        datasets = classification_dataset_names
    else:
        raise NotImplementedError

    log_config = eu.AttrDict(
        directory = eu.DEFAULT_DATA_DIRECTORY,
    )
    summary_logger = log.Logger(log_config)

    for dataset_name in datasets:

        _config = config.copy()

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


        _config.dataset.name = dataset_name

        dataset = eu.misc.create_object_from_config(_config.dataset)
        sample_points, sample_values = dataset.generate_samples()

        if sample_points.shape[-1] > 5:
            print(f"Skipping {dataset_name} because it has more than 5 dimensions")
            continue


        min_vals = np.min(sample_points,axis=0)
        max_vals = np.max(sample_points,axis=0)

        if _config.model.in_features == None:
            _config.model.in_features = sample_points.shape[-1]
            print("Set in_features to ",_config.model.in_features)
        if _config.model.out_features == None:
            _config.model.out_features = len(np.unique(sample_values))
            print("Set out_features to ",_config.model.out_features)


        if None in _config.model.range:
            ### As a test we set the range to the global min and max 
            _config.model.range = (np.min(min_vals,axis=0)*1.2,np.max(max_vals,axis=0)*1.2)
            #_config.model.ranges = list(zip(min_vals,max_vals))
            print("Set ranges to ",_config.model.range)


        if len(_config.model.hidden_features) > 0:
            ### We need to set the first hidden layer to the number of features
            _config.model.hidden_features[0] = _config.model.in_features * _config.model.hidden_features[0]

        model = run_data_classification_experiment(config=_config,logger=logger, **kwargs)

        summary_logger.add_object("dataset",dataset_name)
        #summary_logger.add_object("model",model)

        if isinstance(model, torch.nn.Module):
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            summary_logger.add_value("params",params)

        test_loss = np.array(logger.numpy_data["test_loss"])
        min_test_loss = np.min(test_loss)

        print("Test loss")
        print(test_loss.shape)

        val_loss = np.array(logger.numpy_data["val_loss"])
        min_val_loss_idx = np.argmin(val_loss)

        print("Val loss")
        print(val_loss.shape)

        summary_logger.add_value("min_val_test_loss",test_loss[min_val_loss_idx])
        summary_logger.add_value("min_test_loss",min_test_loss)
        summary_logger.add_value("duration",logger.numpy_data["duration"][-1])
        summary_logger.save()


def run_data_classification_experiment(config=None,logger=None, **kwargs):

    if logger == None:
        logger = log
    

    start_time = time.time()

    # define the default configuration
    default_config = eu.AttrDict(
        model = eu.AttrDict(cls=URBFMLP,            
            in_features=2,
            use_urbf=False,
            range=(-10,5),   
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
            batch_size=64,
            is_classification=True),
        seed = 123,
        test_split_size = 0.2,
        val_split_size = 0.2,
        use_equal_sampling = False,
        log_to_tensorboard = True,
    )

    # set the config based on the default config, given config, and the given function arguments
    config = eu.combine_dicts(kwargs, config, default_config)


    # set random seeds with seed defined in the config
    eu.misc.seed(config)

    ##
    dataset = eu.misc.create_object_from_config(config.dataset)

    sample_points, sample_values = dataset.generate_samples()
    print(f"Sampled {sample_points.shape} {sample_values.shape}")
    
    ## 
    min_vals = np.min(sample_points,axis=0)
    max_vals = np.max(sample_points,axis=0)

    if None in config.model.range:
        ### As a test we set the range to the global min and max 
        config.model.range = [(min_val,max_val) for min_val, max_val in zip(min_vals,max_vals)]
        #config.model.range = (np.min(min_vals,axis=0)*1.2,np.max(max_vals,axis=0)*1.2)
        print(f"Setting range to global min and max: {config.model.range}")

    trainer = eu.misc.create_object_from_config(config.trainer)
    model = eu.misc.create_object_from_config(config.model)
    
    if config.use_equal_sampling:

        res = int(sqrt(sample_values.shape[0]))
        sample_values = np.reshape(sample_values,(res,res,3))
        sample_points = np.reshape(sample_points,(res,res,2))

        train_values = sample_values[::2,::2,:]
        train_points = sample_points[::2,::2,:]

        train_values = train_values.reshape(-1,3)
        train_points = train_points.reshape(-1,2)

        test_values = sample_values[1::2,1::2,:]
        test_points = sample_points[1::2,1::2,:]

        test_values = test_values.reshape(-1,3)
        test_points = test_points.reshape(-1,2)

        val_values = sample_values[::2,1::2,:]
        val_points = sample_points[::2,1::2,:]

        val_values = val_values.reshape(-1,3)
        val_points = val_points.reshape(-1,2)
    else:
        # Split the dataset into training (60%), validation (20%), and test (20%)
        train_points, test_points, train_values, test_values = train_test_split(sample_points, sample_values, test_size=config.test_split_size, random_state=config.seed)
        # Further split the training set into training and validation sets
        train_points, val_points, train_values, val_values = train_test_split(train_points, train_values, test_size= config.val_split_size / (1 - config.test_split_size) , random_state=config.seed)

    train_dataset = FunctionSampleDataset(train_points, train_values)
    val_dataset = FunctionSampleDataset(val_points, val_values)
    test_dataset = FunctionSampleDataset(test_points, test_values)

    model = trainer.train(model,train_dataset,val_dataset=val_dataset,test_dataset=test_dataset,logger=logger)  
    
    end_time = time.time()

    duration = end_time - start_time

    print(f"Duration: {time.strftime('%H:%M:%S', time.gmtime(duration))}s")
    logger.add_value('duration', duration)
    logger.save()

    return model


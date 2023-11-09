from function_regression.function_sample_dataset import FunctionSampleDataset
from function_regression.functions.gaussian_mixture_function import GaussianMixtureFunction
from function_regression.models.urbf_mlp import URBFMLP
import numpy as np
from typing import Any, List,Tuple
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
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
    return samples.T



def run_experiment(config):
    ### here we will first train a given model to fit a given type of function

    print("Starting experiment")

    ranges = [(-5,5),(-5,5)]
    difficulty = 2

    function = GaussianMixtureFunction(2,means=sample_random_arrays(difficulty,ranges),vars=sample_random_arrays(difficulty,[(0.1,3)]))
    
    
    print("Sampling points")
    sample_points, sample_values = function.generate_samples(sample_range=ranges,sample_step=0.01)
    print(f"Got {sample_values.shape}")
    
    function.plot(sample_points, sample_values )   

    # Split the dataset into training (60%), validation (20%), and test (20%)
    train_points, test_points, train_values, test_values = train_test_split(
        sample_points, sample_values, test_size=0.2, random_state=42)

    # Further split the training set into training and validation sets
    train_points, val_points, train_values, val_values = train_test_split(
        train_points, train_values, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2


    #function.plot(train_points,train_values)    
    print(train_points.shape)
    print(test_points.shape)
    print(val_points.shape)



    print("creating model")
    urbf_mlp = URBFMLP(2,1,[8,16,8,4,1])
    urbf_mlp = train(train_points=train_points,train_values=train_values, model=urbf_mlp)



    

    return 0


def train(train_points,train_values,model):
    train_dataset = FunctionSampleDataset(train_points, train_values)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


    print(model.parameters())

    # Define an optimizer and a loss function
    optimizer = torch.optim.SGD(list(model.parameters()), lr=0.01)
    criterion = torch.nn.MSELoss()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    num_epochs = 500  # Number of epochs to train for


    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: compute the model output
            outputs = model(inputs)#.squeeze()

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass: compute the gradient of the loss with respect to model parameters
            loss.backward()

            # Perform a single optimization step (parameter update)
            optimizer.step()

            # Accumulate the running loss
            running_loss += loss.item()

        # Print statistics after every epoch
        print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f}')


    print('Finished Training')

    return model
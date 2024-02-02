import torch
import math
from typing import List,Tuple
import torch.nn as nn
from typing_extensions import Literal

class RBFLayer(torch.nn.Module):

    def __init__(self,
                in_features:int,
                out_features:int,
                data_range:List[Tuple[int]],
                univariate:bool = False,
                learnable:bool = True,
                initial_distribution = "uniform",
                dynamic:bool = False):
        
        super().__init__()
        

        self.univariate = univariate
        self.in_features = in_features
        self.out_features = out_features
        self.range = data_range

        if univariate:
            self.expansion_mapping = torch.zeros((out_features,in_features))

            out_features_per_dim = self.out_features // self.in_features

            for _in_feature in range(self.in_features):
                for _out_feature in range(out_features_per_dim):
                    self.expansion_mapping[_in_feature*(out_features_per_dim) + _out_feature,_in_feature] = 1.0

            self.expansion_mapping.requires_grad_(False)

            ## initialize the means and stds
            means = (torch.zeros(self.out_features))
            stds = (torch.ones(self.out_features))

            if initial_distribution == "uniform":
                for dim, dim_range in enumerate(self.range):
                    dim_min,dim_max = dim_range
                    abs_range = dim_max - dim_min
                    left_features = out_features_per_dim

                    step = abs_range / (left_features)

                    for neuron in range(left_features):

                        means[(dim)*out_features_per_dim + neuron] = dim_min + step*(neuron) + step/2
                        stds[(dim)*out_features_per_dim + neuron] = step*2
            
            elif initial_distribution == "random":

                for dim, dim_range in enumerate(self.range):
                    dim_min,dim_max = dim_range
                    abs_range = dim_max - dim_min
                    left_features = out_features_per_dim

                    step = abs_range / (left_features)

                    for neuron in range(left_features):
                        means[neuron] = torch.rand(1) * (abs_range) + dim_min
                        stds[neuron] = torch.rand(1) * (step) + step

            else:
                raise ValueError(f"initial_distribution must be either uniform or random not {initial_distribution}")

            self.means = torch.nn.Parameter(means)
            self.stds = torch.nn.Parameter(stds)

        else:
            means = torch.rand(self.in_features, self.out_features)

            out_features_per_dim = int(self.out_features ** (1/self.in_features))

            if initial_distribution == "uniform":                
                # Initialize a list to store the range for each dimension
                dimension_ranges = []

                max_step = 0

                for i in range(in_features):
                    # Calculate the step for each dimension
                    step = (self.range[i][1] - self.range[i][0]) / out_features_per_dim

                    if step > max_step:
                        max_step = step

                    # Create a range of values for this dimension
                    dim_range = torch.linspace(self.range[i][0], self.range[i][1], out_features_per_dim)
                    dimension_ranges.append(dim_range)

                # Use meshgrid to create the grid for all dimensions
                grid = torch.meshgrid(*dimension_ranges, indexing='ij')

                # Reshape and store the grid points as means
                _means = torch.stack([g.flatten() for g in grid], dim=1).T

                means[:,:_means.shape[-1]]= _means

                self.means = torch.nn.Parameter(means)
                self.beta = torch.nn.Parameter(torch.ones(self.out_features)* (1 / ( max_step * 2)) ** 2)

            elif initial_distribution == "random":
                abs_range = torch.stack([torch.tensor((self.range[i][1] - self.range[i][0])) for i in range(len(self.range))])
                max_range = abs_range.amax()

                for i in range(self.out_features):
                    for j in range(self.in_features):
                        means[j,i] = torch.rand(1) * (self.range[j][1] - self.range[j][0]) + self.range[j][0]

                self.means = torch.nn.Parameter(means)
                self.beta = torch.nn.Parameter(torch.ones(self.out_features)* (1 / ( max_range * (torch.rand(1) * 3 + 1))) ** 2)

            else:
                raise ValueError(f"initial_distribution must be either uniform or random not {initial_distribution}")


        if dynamic:
            self.linear_layer_grad_output = None


        if not learnable:
            self.means.requires_grad_(False)
            if univariate:
                self.stds.requires_grad_(False)
            else:
                self.beta.requires_grad_(False)

    def liner_backward_hook(self,module, grad_input, grad_output):
        print(f"linear backward hook grad_output: {grad_output[0].shape}")
        self.linear_layer_grad_output = grad_output[0]


    def forward(self,x):
        if self.univariate:
            x = (x @ self.expansion_mapping.T - self.means) / self.stds
            x = torch.exp(-0.5 * x ** 2)
        else:
            x = x.unsqueeze(-1).repeat(1,1,self.out_features) ### spread the input over the output dimension
            x = (x - self.means[None,:,:]).norm(dim=1)
            x = torch.exp(-0.5 * self.beta[None,:] * x ** 2)
        return x

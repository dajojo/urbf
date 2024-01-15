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
                univariate:bool = False):
        
        super().__init__()
        

        self.univariate = univariate
        self.in_features = in_features
        self.out_features = out_features
        self.range = data_range

        
        if univariate:
            self.expansion_mapping = torch.zeros((out_features,in_features))#nn.Linear(in_features,out_features,bias=False)

            out_features_per_dim = self.out_features // self.in_features

            for _in_feature in range(self.in_features):
                for _out_feature in range(out_features_per_dim):
                    self.expansion_mapping[_in_feature*(out_features_per_dim) + _out_feature,_in_feature] = 1.0

            self.expansion_mapping.requires_grad_(False)

            ## initialize the means and stds
            means = (torch.zeros(self.out_features))
            stds = (torch.ones(self.out_features))

            for dim, dim_range in enumerate(self.range):
                dim_min,dim_max = dim_range
                abs_range = dim_max - dim_min
                left_features = out_features_per_dim

                step = abs_range / (left_features)

                for neuron in range(left_features):

                    means[(dim)*out_features_per_dim + neuron] = dim_min + step*(neuron) + step/2
                    stds[(dim)*out_features_per_dim + neuron] = step*2

            self.means = torch.nn.Parameter(means)
            self.stds = torch.nn.Parameter(stds)

        else:
            means = torch.zeros(self.in_features, self.out_features)

            step = torch.stack([torch.tensor((self.range[i][1] - self.range[i][0]) / self.out_features) for i in range(len(self.range))])
            for i in range(self.out_features):
                for j in range(self.in_features):
                    means[j,i] = self.range[j][0] + i * step[j]


            max_step = step.amax()

            self.means = torch.nn.Parameter(means)
            self.beta = torch.nn.Parameter(torch.ones(self.out_features) * max_step * 2)
            self.alpha = torch.nn.Parameter(torch.ones(self.out_features))

    def forward(self,x):
        if self.univariate:
            x = (x @ self.expansion_mapping.T - self.means) / self.stds
            x = torch.exp(-0.5 * x ** 2)
        else:
            x = x.unsqueeze(-1).repeat(1,1,self.out_features) ### spread the input over the output dimension
            x = (x - self.means[None,:,:]).norm(dim=1)
            x = x * self.beta[None,:]
            
            x = torch.exp(-0.5 * x ** 2)
            x = x * self.alpha[None,:]

        return x

import torch
import math
from typing import List,Tuple

class URBFLayer(torch.nn.Module):
    def __init__(self,in_features:int,out_features:int,ranges:List[Tuple[int]]):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.ranges = ranges

        assert len(self.ranges) == self.in_features, "Number of range pairs must match the number of input features"
        assert (self.out_features % self.in_features) == 0, "out_features must be a multiple of in_features"


        self.means = torch.nn.Parameter(torch.zeros(self.out_features))
        self.vars = torch.nn.Parameter(torch.ones(self.out_features))


        means = torch.zeros_like(self.means)
        for dim, dim_range in enumerate(self.ranges):
            dim_min,dim_max = dim_range

            step = (dim_max - dim_min)/(self.out_features_per_in_feature - 1)
            
            for out_feat_dim in range(self.out_features_per_in_feature):
                means[dim*self.out_features_per_in_feature + out_feat_dim] = dim_min + step*out_feat_dim
        
        self.means = torch.nn.Parameter(means)


    @property
    def out_features_per_in_feature(self) -> int:
        return self.out_features // self.in_features

    def forward(self,x):
        """
        Computes the ouput of the URBF layer given an input vector

        Parameters
        ----------
            input: torch.Tensor
                Input tensor of size B x F, where B is the batch size,
                and F is the feature space dimensionality of the input

        Returns
        ----------
            out: torch.Tensor
                Output tensor of size B x Fout, where B is the batch
                size of the input, and Fout is the output feature space
                dimensionality
        """
        
        
        # reapeat input vector so that every map-neuron gets its accordingly input
        # example: n_neuron_per_inpu = 3 then [[1,2,3]] --> [[1,1,1,2,2,2,3,3,3]]
        x = x.repeat_interleave(repeats=self.out_features_per_in_feature, dim=-1)

        # calculate gauss activation per map-neuron
        return torch.exp(-0.5 * ((x - self.means) / self.vars) ** 2)
    
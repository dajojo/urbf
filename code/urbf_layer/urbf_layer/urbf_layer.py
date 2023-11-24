import torch
import math
from typing import List,Tuple
import torch.nn as nn

class URBFLayer(torch.nn.Module):
    def __init__(self,in_features:int,out_features:int,ranges:List[Tuple[int]],use_split_merge=True,split_merge_temperature=1/10):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.ranges = ranges

        self.use_split_merge = use_split_merge
        self.split_merge_temperature = torch.ones(self.in_features) * split_merge_temperature

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

        self.register_backward_hook(self.backward_hook)

        self.grads = []
        self.means_hist = []
        self.acc_grad = torch.zeros_like(self.means)
        self.acc_grad_hist = []

    @property
    def out_features_per_in_feature(self) -> int:
        return self.out_features // self.in_features


    def backward_hook(self, module, grad_input, grad_output):
        # Your custom logic here
        print("Custom layer backward hook called")
        print(f"input: {grad_input[0].shape} output: {grad_output[0].shape}")
        # Example: print gradients
        print("Gradients at this layer:", grad_output)

        self.grads.append(grad_input[0])
        self.acc_grad = self.acc_grad + grad_input[0].abs().sum(dim=0).detach().clone()


        if self.use_split_merge:
            self.check_for_split_merge()

        self.means_hist.append(self.means.detach().clone())
        self.acc_grad_hist.append(self.acc_grad)


    def check_for_split_merge(self):
        #### check_for_split_merge
        #   Currently the outcome is dependent on the BatchSize since max 1 split & merge per input neuron is performed after every batch
        #   Should we enable to perform multiple Split & Merge operations per backward pass per input neuron?
        #   But then, we might run into issues since too much is happening?
        #   We also need a way to update the accumulated grad values..



        print("check_for_split_merge")

        for in_feature in range(self.in_features):
            out_feats = self.acc_grad[in_feature*self.out_features_per_in_feature:(in_feature+1)*self.out_features_per_in_feature]

            max_val,max_idx = out_feats.max(dim=0)
            min_val,min_idx = out_feats.min(dim=0)
            
            if max_val*self.split_merge_temperature[in_feature] > min_val:
                # Perform Split and Merge!
                self.split_merge_temperature[in_feature] = self.split_merge_temperature[in_feature]/2 #### should we treat it as a 'Temperature' which is cooling down to reduce rearrangements later during training?

                max_mean = self.means[max_idx + in_feature*self.out_features_per_in_feature]
                self.means[min_idx + in_feature*self.out_features_per_in_feature] = max_mean - self.vars[max_idx + in_feature*self.out_features_per_in_feature]/2 #self.means[min_idx + in_feature*self.out_features_per_in_feature] + 
                self.means[max_idx + in_feature*self.out_features_per_in_feature] = max_mean + self.vars[max_idx + in_feature*self.out_features_per_in_feature]/2 

                self.acc_grad[in_feature*self.out_features_per_in_feature:(in_feature+1)*self.out_features_per_in_feature] = torch.zeros(self.out_features_per_in_feature)


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
    
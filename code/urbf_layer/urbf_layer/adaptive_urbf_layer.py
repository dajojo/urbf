import torch
import math
from typing import List,Tuple
import torch.nn as nn
from typing_extensions import Literal

class AdaptiveURBFLayer(torch.nn.Module):

    def __init__(self,
                in_features:int,
                out_features:int):
        super().__init__()

        print("init AdaptiveURBFLayer")

        self.in_features = in_features
        self.out_features = out_features

        self.rbf_layer = AdaptiveRBFLayer(self.out_features)
        self.rbf_layer.register_full_backward_hook(self.rbf_backward_hook)

        self.expansion_mapping = torch.zeros((self.out_features,self.in_features))

        ### init the expansion mapping to equal mapping...
        for in_feature in range(self.in_features):
            for _out_feature in range(self.out_features//self.in_features):
                self.expansion_mapping[in_feature*(self.out_features//self.in_features) + _out_feature,in_feature] = 1
         
        print(f"init expansion mapping: {self.expansion_mapping}")
        self.expansion_mapping = torch.nn.Parameter(self.expansion_mapping)


    def rbf_backward_hook(self,module, grad_input, grad_output):
        print("---")



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
        x.requires_grad = True

        if self.training:
            min = x.amin(dim=0)
            max = x.amax(dim=0)
                        
            input_batch_range = torch.stack([min,max],dim=1)

            if self.rbf_layer.adaptive_range == None:
                ### initialize adaptive range
                self.rbf_layer.adaptive_range = input_batch_range
            else:
                delta = input_batch_range - self.rbf_layer.adaptive_range

                if (delta[:,0] < 0).any() or (delta[:,1] > 0).any():
                    self.rbf_layer.update_adaptive_range(self.expansion_mapping,delta)


        ### Expand the dimensionality
        x = (self.expansion_mapping @ x.transpose(1,0)).transpose(1,0)
         
        # calculate gauss activation per map-neuron
        return self.rbf_layer(x)




class AdaptiveRBFLayer(torch.nn.Module):
        
    def __init__(self,n_features) -> None:
        super().__init__()
        print("Init AdaptiveRBFLayer")
        self.n_features = n_features

        self.means = torch.nn.Parameter(torch.zeros(self.n_features))
        self.vars = torch.nn.Parameter(torch.ones(self.n_features))
        self.coefs = torch.nn.Parameter(torch.ones(self.n_features))
        
        self.adaptive_range = None

    def update_adaptive_range(self,expansion_mapping,delta_range):
        print("updating adaptive range")

        #### 
        delta_range[:,0][delta_range[:,0] > 0] = 0
        delta_range[:,1][delta_range[:,1] < 0] = 0

        with torch.no_grad():

            #self.adaptive_range = self.adaptive_range + delta_range
            #print(f"New adaptive range: {self.adaptive_range}")

            for dim, dim_range in enumerate(delta_range):
                delta_dim_min,delta_dim_max = dim_range

                #### We take the neuron with min var of the targeted mapping

                #expansion_mapping[:,dim] # -> [out] indicating if a neuron is connected to dim

                self.vars[expansion_mapping[:,dim] > 0] 


    def forward(self,x):
        ### B x C
        print(f"AdaptiveRBFLayer: {x.shape}")
        return torch.exp(-0.5 * ((x - self.means) / self.vars) ** 2) * self.coefs
    
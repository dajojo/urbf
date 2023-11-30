import torch
import math
from typing import List,Tuple
import torch.nn as nn
from typing_extensions import Literal

class URBFLayer(torch.nn.Module):


    def __init__(self,
                 in_features:int,
                 out_features:int,
                 ranges:List[Tuple[int]],
                 use_split_merge=True,
                 split_merge_temperature=1/10,
                 use_back_tray=False,
                 back_tray_ratio = 0.5,
                 grad_signal: Literal["input","output","mean"] = "input" ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.ranges = ranges

        self.grad_signal = grad_signal

        self.use_split_merge = use_split_merge
        self.split_merge_temperature = torch.ones(self.in_features) * split_merge_temperature

        self.use_back_tray = use_back_tray

        assert len(self.ranges) == self.in_features, "Number of range pairs must match the number of input features"
        assert (self.out_features % self.in_features) == 0, "out_features must be a multiple of in_features"

        assert not (use_back_tray and use_split_merge), "Split and Merge and Backtray can not be used at the same time."

        self.rbf_layer = RBFLayer(self.out_features,self.ranges,out_features_per_in_feature=self.out_features_per_in_feature,use_back_tray=use_back_tray,back_tray_ratio=back_tray_ratio)
        self.rbf_layer.register_full_backward_hook(self.backward_hook)


        self.grads = []
        self.means_hist = []
        self.acc_grad = torch.zeros_like(self.rbf_layer.means)
        self.acc_grad_hist = []


    @property
    def out_features_per_in_feature(self) -> int:
        return self.out_features // self.in_features


    def backward_hook(self,module, grad_input, grad_output):
        # Your custom logic here
        #grad_signal = self.rbf_layer.means.grad.unsqueeze(0)#grad_input[0]#self.rbf_layer.means.grad

        if self.grad_signal == 'input':
            grad_signal = grad_input[0]
        elif self.grad_signal == 'output':
            grad_signal = grad_output[0]
        elif self.grad_signal == 'mean':
            grad_signal = self.rbf_layer.means.grad.unsqueeze(0)
        else:
            raise f"Unexpected grad signal: {self.grad_signal}"


        self.grads.append(grad_signal)
        self.acc_grad = self.acc_grad + grad_signal.abs().sum(dim=0).detach().clone()#grad_input[0].abs().sum(dim=0).detach().clone()

        if self.use_split_merge:
            self.check_for_split_merge()

        elif self.use_back_tray:
            self.check_for_add_from_back_tray()

        self.means_hist.append(self.rbf_layer.means.detach().clone())
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

                max_mean = self.rbf_layer.means[max_idx + in_feature*self.out_features_per_in_feature]
                self.rbf_layer.means[min_idx + in_feature*self.out_features_per_in_feature] = max_mean - self.rbf_layer.vars[max_idx + in_feature*self.out_features_per_in_feature]/2 #self.means[min_idx + in_feature*self.out_features_per_in_feature] + 
                self.rbf_layer.means[max_idx + in_feature*self.out_features_per_in_feature] = max_mean + self.rbf_layer.vars[max_idx + in_feature*self.out_features_per_in_feature]/2 

                self.acc_grad[in_feature*self.out_features_per_in_feature:(in_feature+1)*self.out_features_per_in_feature] = torch.zeros(self.out_features_per_in_feature)


    def check_for_add_from_back_tray(self):

        ### Idea: Use only a part of the Gaussians and keep the rest in a 'Back Tray'
        # similar to split and merge, find a condition under which we introduce neurons to areas where it is needed.
        # Without altering existing neurons, we enable a new neuron and add it to the area which is affected by high grads  
        # ...TODO
        # - How can we control


        print("check_for_add_from_back_tray")

        for in_feature in range(self.in_features):
            out_feats_acc_grad = self.acc_grad[in_feature*self.rbf_layer.out_features_per_in_feature:in_feature*self.rbf_layer.out_features_per_in_feature+self.rbf_layer.active_out_features_per_in_feature[in_feature]]

            max_val,max_idx = out_feats_acc_grad.max(dim=0)
            min_val,min_idx = out_feats_acc_grad.min(dim=0)
            
            if max_val*self.split_merge_temperature[in_feature] > min_val and self.rbf_layer.active_out_features_per_in_feature[in_feature] < self.out_features_per_in_feature:
                print("adding from backtray!")
                # Perform Split and Merge!
                self.split_merge_temperature[in_feature] = self.split_merge_temperature[in_feature]/2 #### should we treat it as a 'Temperature' which is cooling down to reduce rearrangements later during training?

                max_mean = self.rbf_layer.means[max_idx + in_feature*self.out_features_per_in_feature]
                max_var = self.rbf_layer.vars[max_idx + in_feature*self.out_features_per_in_feature]


                #self.rbf_layer.means[min_idx + in_feature*self.out_features_per_in_feature] = max_mean - self.rbf_layer.vars[max_idx + in_feature*self.out_features_per_in_feature]/2 #self.means[min_idx + in_feature*self.out_features_per_in_feature] + 

                
                new_gauss_idx = self.rbf_layer.active_out_features_per_in_feature[in_feature] + in_feature*self.out_features_per_in_feature

                if max_idx == 0:
                    self.rbf_layer.means[new_gauss_idx] = max_mean + max_var/2
                elif max_idx == len(out_feats_acc_grad) - 1:
                    self.rbf_layer.means[new_gauss_idx] = max_mean - max_var/2
                else:
                    if out_feats_acc_grad[max_idx - 1] < out_feats_acc_grad[max_idx + 1] and self.rbf_layer.means[max_idx - 1 ] < self.rbf_layer.means[max_idx + 1 ]:
                        self.rbf_layer.means[new_gauss_idx] = max_mean + max_var/2
                    else:
                        self.rbf_layer.means[new_gauss_idx] = max_mean - max_var/2

                self.rbf_layer.active[new_gauss_idx] = True
                self.rbf_layer.active_out_features_per_in_feature[in_feature] = self.rbf_layer.active_out_features_per_in_feature[in_feature] + 1

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

        x.requires_grad = True          # We use this to make the hook return an input value

        # reapeat input vector so that every map-neuron gets its accordingly input
        # example: n_neuron_per_inpu = 3 then [[1,2,3]] --> [[1,1,1,2,2,2,3,3,3]]
        x = x.repeat_interleave(repeats=self.out_features_per_in_feature, dim=-1)

        # calculate gauss activation per map-neuron
        return self.rbf_layer(x)

    


class RBFLayer(torch.nn.Module):
        
    def __init__(self,n_features,
                 init_ranges,
                 out_features_per_in_feature,                 
                 use_back_tray=False,
                 back_tray_ratio = 0.5,) -> None:
        super().__init__()

        
        self.n_features = n_features
        self.init_ranges = init_ranges

        self.out_features_per_in_feature = out_features_per_in_feature

        self.use_back_tray = use_back_tray
        self.back_tray_ratio = back_tray_ratio

        self.means = torch.nn.Parameter(torch.zeros(self.n_features))
        self.vars = torch.nn.Parameter(torch.ones(self.n_features))
        self.coefs = torch.nn.Parameter(torch.ones(self.n_features))

        self.active = torch.nn.Parameter(torch.zeros(self.n_features).bool(),requires_grad=False)

        if use_back_tray:
            print("using backtray...")
            self.active_out_features_per_in_feature = (torch.ones(len(self.init_ranges))* int(out_features_per_in_feature*back_tray_ratio)).to(torch.int)
        else:
            self.active_out_features_per_in_feature = (torch.ones(len(self.init_ranges)) * out_features_per_in_feature).to(torch.int)


        means = torch.zeros_like(self.means)
        vars =  torch.zeros_like(self.vars)

        for dim, dim_range in enumerate(self.init_ranges):
            dim_min,dim_max = dim_range
            
            abs_range = dim_max - dim_min

            level = 1
            left_features = self.out_features_per_in_feature

            while left_features > 0:
                for neuron in range(level):
                    means[(dim + 1)*self.out_features_per_in_feature - left_features] = dim_min + (abs_range/(level*2))*(neuron*2 + 1)
                    vars[(dim + 1)*self.out_features_per_in_feature - left_features] = abs_range/(level * 2)

                    self.active[(dim + 1)*self.out_features_per_in_feature - left_features] = True

                    left_features = left_features - 1

                level = level + 1

                print(f"Entering level {level} with left features {left_features}")

            # step = (dim_max - dim_min)/(self.active_out_features_per_in_feature[dim] - 1)
            # for out_feat_dim in range(self.active_out_features_per_in_feature[dim]):
            #    means[dim*self.out_features_per_in_feature + out_feat_dim] = dim_min + step*out_feat_dim
            #    vars[dim*self.out_features_per_in_feature + out_feat_dim] = step/2
               
            #    self.active[dim*self.out_features_per_in_feature + out_feat_dim] = True
        
        self.means = torch.nn.Parameter(means)
        self.vars = torch.nn.Parameter(vars)


    def forward(self,x):
        # calculate gauss activation per map-neuron

        if self.use_back_tray:
            print(f"deactivating... {x.shape}")
            x = x * self.active[None,:]

        return torch.exp(-0.5 * ((x - self.means) / self.vars) ** 2) * self.coefs
    
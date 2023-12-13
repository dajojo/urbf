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

        self.input = None

        self.in_features = in_features
        self.out_features = out_features

        self.rbf_layer = AdaptiveRBFLayer(self.out_features)
        self.linear_layer = torch.nn.Linear(self.out_features,self.out_features)
        self.activation = torch.nn.ReLU()

        ### init the expansion mapping to equal mapping...
        self.expansion_mapping = torch.zeros((self.out_features,self.in_features))
        for in_feature in range(self.in_features):
            for _out_feature in range(self.out_features//self.in_features):
                self.expansion_mapping[in_feature*(self.out_features//self.in_features) + _out_feature,in_feature] = 1
        
        ### Init adaptive range
        self.adaptive_range = None

        ### Significance
        self.significance = torch.ones(self.out_features)*0.5

        #self.rbf_layer.register_full_backward_hook(self.rbf_backward_hook)
        self.linear_layer.register_full_backward_hook(self.liner_backward_hook)
        #self.register_full_backward_hook(self.backwad_hook)

        ### keep track of the loss regarding linear layer output
        self.linear_layer_grad_output = None
    

    def backwad_hook(self,module, grad_input, grad_output):
        print("Backward hook")
 

    def liner_backward_hook(self,module, grad_input, grad_output):
        #print(f"LinearLayer backward hook: input->{grad_input[0].shape} output->{grad_output[0].shape}")
        if self.training:
            self.linear_layer_grad_output = grad_output[0]


    def rbf_backward_hook(self,module, grad_input, grad_output):
        print(f"AdaptiveURBFLayer backward hook: input->{grad_input[0].shape} output->{grad_output[0].shape}")


    # def init_spektrum_pattern(self,adaptive_range):
    #     means = torch.zeros_like(self.rbf_layer.means)
    #     vars =  torch.zeros_like(self.rbf_layer.vars)

    #     for dim, dim_range in enumerate(adaptive_range):

    #         dim_min,dim_max = dim_range
            
    #         abs_range = dim_max - dim_min

    #         level = 1
    #         left_features = self.out_features//self.in_features
    #         while left_features > 0:
    #             for neuron in range(level):

    #                 if left_features == 0:
    #                     break

    #             means[(dim + 1)*(self.out_features//self.in_features) - left_features] = dim_min + (abs_range/(level*2))*(neuron*2 + 1)
    #             vars[(dim + 1)*(self.out_features//self.in_features) - left_features] = abs_range/(level * 2)

    #             left_features = left_features - 1

    #         level = level + 1

    #     self.rbf_layer.means = torch.nn.Parameter(means)
    #     self.rbf_layer.vars = torch.nn.Parameter(vars)


    def prune(self):
        prune_threshold = 0.01### make it dependent on the absolute value of the adaptive range: 
        values, indices = self.significance.topk(self.out_features // 10,largest=False)

        filtered_indices = indices[values < prune_threshold]
        
        self.expansion_mapping[filtered_indices,:] = 0
        self.rbf_layer.means[filtered_indices].data.fill_(0)
        self.rbf_layer.vars[filtered_indices].data.fill_(1)

        self.significance[filtered_indices] = 0.5
        ### Significance is set 0.5 to avoid pruning the same neurons again and again

        self.linear_layer.weight[:,filtered_indices].data.fill_(0.01)

    def grow(self):
        ## implement the grow algorithm as described in algorithm 1
        ## The bridgin gradient matrix can be calculated using "virtual" inputs coming from several gaussian candidates.
        ## we can pick the most promising candidates... also considering the gaussians that are indicated by "unprocessed" inputs
        ## The algorithm is as follows:
        #### 1. Extract all free slots
        #### 2. create the prototype gaussians
        #### 3. calculate the output of the gaussians
        #### 4. calculate the connection growth (Based on hebbian theory)

        self.input = self.input.detach() ## B x F

        input_activations = []
        input_dim_rbf_layers = []

        for in_dim in range(self.in_features):
            dim_min,dim_max = self.adaptive_range[in_dim]
            abs_range = dim_max - dim_min

            level = 1

            total_features = self.out_features * 2
            left_features = total_features

            means = torch.zeros(total_features)
            vars = torch.ones(total_features)

            while left_features > 0:
                for neuron in range(level):

                    if left_features == 0:
                        break

                    means[total_features - left_features] = dim_min + (abs_range/(level*2))*(neuron*2 + 1)
                    vars[total_features - left_features] = abs_range/(level * 2)

                    left_features = left_features - 1

                level = level + 1

            input_dim_rbf_layer = AdaptiveRBFLayer(total_features,means=means,vars=vars)
            act = input_dim_rbf_layer(torch.ones((self.input.shape[0],total_features))*self.input[:,in_dim][:,None])
            input_dim_rbf_layers.append(input_dim_rbf_layer)
            input_activations.append(act)

        input_activations = torch.stack(input_activations,dim=1) ## B x in x prototypes
        ### linear_layer_grad_output: B x C

        linear_layer_grad_output_corr = (self.linear_layer_grad_output[:,None,None,:] * input_activations[:,:,:,None]).mean(dim=0) ## B x in x prototypes x next_neuron -> C x F

        slot_indices = (self.expansion_mapping.sum(dim=-1) == 0).nonzero()#.squeeze() ## C

        if slot_indices.shape[0] >= 0:
            for slot_idx in slot_indices:

                # Find the maximum value and its index in the tensor
                max_val, max_in_idx = torch.max(linear_layer_grad_output_corr, dim=0)

                # Find the maximum index in the second dimension
                _, max_prototype_idx = torch.max(max_val, dim=0)

                # Find the maximum index in the third dimension
                _, max_neuron_idx = torch.max(max_prototype_idx, dim=0)

                max_prototype_idx = max_prototype_idx[max_neuron_idx]
                max_in_idx = max_in_idx[max_prototype_idx, max_neuron_idx]

                mean = input_dim_rbf_layers[max_in_idx].means[max_prototype_idx]
                var = input_dim_rbf_layers[max_in_idx].vars[max_prototype_idx]

                self.rbf_layer.means[slot_idx].data = mean
                self.rbf_layer.vars[slot_idx].data = var

                self.expansion_mapping[slot_idx,max_in_idx] = 0.5

                print(f"Growing at: {slot_idx} set weight from: {self.linear_layer.weight[max_neuron_idx,slot_idx]} -> {linear_layer_grad_output_corr[max_in_idx,max_prototype_idx,max_neuron_idx]}")

                ### we need to update the linear layer weights
                self.linear_layer.weight[max_neuron_idx,slot_idx].data.fill_(linear_layer_grad_output_corr[max_in_idx,max_prototype_idx,max_neuron_idx] * 0.01)

                linear_layer_grad_output_corr[max_in_idx,max_prototype_idx,:] = 0


        self.linear_layer_grad_output = None
        self.input = None


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

        if self.training:
            x.requires_grad = True

            ### verify adaptive range
            curr_range = torch.stack([x.amin(dim=0),x.amax(dim=0)],dim=1)
            #print(f"adaptive range: {self.adaptive_range} from current range: {curr_range}",)
            if self.adaptive_range == None:
                #### init the adaptive range
                self.adaptive_range = curr_range
            else:    
                delta_range = curr_range - self.adaptive_range            
                delta_range[:,0][delta_range[:,0] > 0] = 0
                delta_range[:,1][delta_range[:,1] < 0] = 0
                if delta_range.abs().sum() > 0:
                    self.adaptive_range = self.adaptive_range + delta_range
                    #self.rbf_layer.init_spektrum_pattern(self.adaptive_range)
                    print(f"new adaptive range: {self.adaptive_range} from current range: {curr_range}",)

            self.prune()
            if (self.expansion_mapping.sum(dim=-1) == 0).any() and self.linear_layer_grad_output != None and self.input != None:
               self.grow()

            self.input = x

        ### Expand the dimensionality
        x = (self.expansion_mapping @ x.transpose(1,0)).transpose(1,0)
         
        ## calculate gauss activation per map-neuron
        y = self.rbf_layer(x)

        ## store significance
        self.significance = 1/2*(self.significance + y.mean(dim=0))

        y = self.linear_layer(y)

        #### Hoook in here to get dL/du (presynaptic gradient) ??
        y = self.activation(y)        

        return y




class AdaptiveRBFLayer(torch.nn.Module):
        
    def __init__(self,n_features, means = None, vars = None) -> None:
        super().__init__()
        #print("Init AdaptiveRBFLayer")
        self.n_features = n_features

        if means == None:
            means = torch.zeros(self.n_features)
        if vars == None:
            vars = torch.ones(self.n_features)

        self.means = torch.nn.Parameter(means)
        self.vars = torch.nn.Parameter(vars)


    def init_spektrum_pattern(self,adaptive_range):

        means = torch.zeros_like(self.means)
        vars =  torch.zeros_like(self.vars)

        self.out_features_per_in_feature = self.n_features//len(adaptive_range)

        for dim, dim_range in enumerate(adaptive_range):

            dim_min,dim_max = dim_range
            
            abs_range = dim_max - dim_min

            level = 1
            left_features = self.out_features_per_in_feature
            while left_features > 0:
                for neuron in range(level):

                    if left_features == 0:
                        break

                means[(dim + 1)*self.out_features_per_in_feature - left_features] = dim_min + (abs_range/(level*2))*(neuron*2 + 1)
                vars[(dim + 1)*self.out_features_per_in_feature - left_features] = abs_range/(level * 2)

                left_features = left_features - 1

            level = level + 1

        self.means.data = means
        self.vars.data = vars


    def forward(self,x):
        ### B x C

        y = torch.exp(-0.5 * ((x - self.means) / self.vars) ** 2)
        #### Reintroduce the scaling factor???
        y = y * 1 / (self.vars.abs() * math.sqrt(2 * math.pi))

        return y
    
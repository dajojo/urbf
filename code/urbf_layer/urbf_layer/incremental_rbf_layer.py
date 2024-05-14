

from typing import List, Tuple
import torch

###
# 1. Full Cov Matrix -> Rotated Ellipsis (C is K x K)
# -> x = exp(- (x - b).T @ C @  (x - b) )
# 2. Diagonal Cov Matrix -> Axis Aligned Ellipsis (C is K x K but with zeros on the off-diagonals)
# -> x = exp(- (x - b).T @ C @  (x - b) )
# 3. Scalar -> Circle
# -> x = exp(- (x - b)^2/(2c^2) )
# 4. Univariate -> Line
# -> x = exp(- (x - b)^2/(2c^2) )


## complexity: full, diagonal, scalar, univariate

class IncrementalRBFLayer(torch.nn.Module):
    def __init__(self,
                in_features:int,
                out_features:int,
                data_range:List[Tuple[int]],
                complexity: str = "full",):
        
        super().__init__()

        assert complexity in ["full","diagonal","scalar","univariate"], "Complexity must be one of 'full', 'diagonal', 'scalar', 'univariate'"

        self.complexity = complexity


        self.in_features = in_features
        self.out_features = out_features
        self.range = data_range

        if complexity == "univariate":
            self.expansion_mapping = torch.zeros((out_features,in_features))

            out_features_per_dim = self.out_features // self.in_features

            for _in_feature in range(self.in_features):
                for _out_feature in range(out_features_per_dim):
                    self.expansion_mapping[_in_feature*(out_features_per_dim) + _out_feature,_in_feature] = 1.0

            self.expansion_mapping.requires_grad_(False)

            ## initialize the means and stds
            means = (torch.zeros(self.out_features))
            vars = (torch.ones(self.out_features))

            for dim, dim_range in enumerate(self.range):
                dim_min,dim_max = dim_range
                abs_range = dim_max - dim_min
                left_features = out_features_per_dim

                step = abs_range / (left_features)

                for neuron in range(left_features):

                    means[(dim)*out_features_per_dim + neuron] = dim_min + step*(neuron) + step/2
                    vars[(dim)*out_features_per_dim + neuron] = step*2
        

            vars = torch.div(torch.ones(self.out_features),vars)

            self.means = torch.nn.Parameter(means)
            self.vars = torch.nn.Parameter(vars)

        else:
            means = torch.rand(self.in_features, self.out_features)

            out_features_per_dim = int(self.out_features ** (1/self.in_features))

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

            print("max step: ", max_step)   

            # Use meshgrid to create the grid for all dimensions
            grid = torch.meshgrid(*dimension_ranges, indexing='ij')

            # Reshape and store the grid points as means
            _means = torch.stack([g.flatten() for g in grid], dim=1).T

            means[:,:_means.shape[-1]]= _means

            self.means = torch.nn.Parameter(means)

            if complexity == "full":
                self.vars = torch.nn.Parameter(((torch.zeros(self.out_features,self.in_features,self.in_features) + torch.eye(self.in_features)) * max_step).inverse())
            elif complexity == "diagonal":
                self.vars = torch.nn.Parameter(torch.div(torch.ones(self.out_features,self.in_features),(torch.ones(self.out_features,self.in_features) * max_step)))
            elif complexity == "scalar":
                self.vars = torch.nn.Parameter((torch.ones(self.out_features)* 1/max_step))
            

    # def liner_backward_hook(self,module, grad_input, grad_output):
    #     print(f"linear backward hook grad_output: {grad_output[0].shape}")
    #     self.linear_layer_grad_output = grad_output[0]


    def forward(self,x):

        if self.complexity == "univariate":
            x = x @ self.expansion_mapping.T
            x = (x - self.means)

            x = x ** 2 * self.vars

        else:
            x = x.unsqueeze(-1).repeat(1,1,self.out_features)
            x = (x - self.means[None,:,:]).transpose(1,2)

            
            if self.complexity == "full":
                x = x.unsqueeze(-2) @ self.vars[None,:,:,:] @ x.unsqueeze(-1)
            elif self.complexity == "diagonal":
                x = x.unsqueeze(-2) @ torch.diag_embed(self.vars)[None,:,:,:] @ x.unsqueeze(-1)
            elif self.complexity == "scalar":
                x = (x*x).sum(dim=-1) * self.vars

        x = torch.exp(-0.5 * x)

        return x.squeeze()
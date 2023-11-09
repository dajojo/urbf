from typing import Any, List,Tuple
import matplotlib.pyplot as plt
import torch
from urbf_layer.urbf_layer import URBFLayer

class URBFMLP(torch.nn.Module):

    def __init__(self, in_features:int, out_features:int, hidden_features:List[int]):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.hidden_features = hidden_features

        self.layers = []

        self.layers.append(URBFLayer(in_features=in_features,out_features=hidden_features[0]))
        self.layers.append(torch.nn.ReLU())


        in_dim = hidden_features[0]
        for hidden_dim in hidden_features[:-1]:
            self.layers.append(torch.nn.Linear(in_dim, hidden_dim))
            self.layers.append(torch.nn.ReLU())
            in_dim = hidden_dim
    
        self.layers.append(torch.nn.Linear(in_dim, hidden_features[-1]))

        self.layers = torch.nn.Sequential(*self.layers)


    def forward(self,x):
        
        return self.layers(x)




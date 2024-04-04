import torch
import torchvision
from torch import nn
from typing import List, Union
import torch.nn.functional as F
import math

from emlp.nn.pytorch import EMLP, MLP as EMLP_MLP

from model.networks.linear import LinearP

class MLP(torch.nn.Module):
    def __init__(self, 
            in_features: int,
            bias: bool,
            device: str,
            activation = torch.nn.ReLU,
            linearmodules: List[Union[LinearP, nn.Linear]] = [LinearP, nn.Linear],
            hidden_implicit_layers: List[int] = None,
            n_hidden_layers = 1,
            out_features: int = 1,
            P_init: Union[torch.Tensor, str] = 'none',
            train_weights = True, 
            train_P = False,
        ):
        super().__init__()

        linear_kwargs = dict(
            in_features=in_features, out_features=in_features, bias=bias, device=device,
            P_init = P_init, train_weights=train_weights, train_P=train_P,
        )
        if hidden_implicit_layers is not None:
            linear_kwargs['hidden_implicit_layers'] = hidden_implicit_layers

        layers = []
        for _ in range(n_hidden_layers):
            layers.append(linearmodules[0]( **linear_kwargs ))
                # in_features=in_features, out_features=in_features, bias=bias, device=device,
                # P_init = P_init, train_weights=train_weights, train_P=train_P,
                # hidden_implicit_layers = hidden_implicit_layers,
            # ))
            layers.append(activation())

        layers.append(linearmodules[1](in_features=in_features, out_features=out_features, bias = bias))

        self.mlp = torch.nn.Sequential(*layers)

        # self.mlp = torch.nn.Sequential(
        #     linearmodules[0](
        #         in_features=in_features, out_features=in_features, bias=bias, device=device,
        #         P_init = P_init, train_weights=train_weights, train_P=train_P
        #     ),
        #     activation(),

        #     linearmodules[0](
        #         in_features=in_features, out_features=in_features, bias=bias, device=device,
        #         P_init = P_init, train_weights=train_weights, train_P=train_P
        #     ),
        #     activation(),

        #     linearmodules[1](in_features=in_features, out_features=out_features, bias = bias),
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return x
    
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, x):
        '''
        Reshapes the input x according to the shape saved in the view data structure.
        '''
        out = x.view(*self.shape)
        return out
    
def MLPBlock(cin,cout):
    return nn.Sequential(nn.Linear(cin,cout),nn.ReLU())



        
class CombiMLP(torch.nn.Module):
    def __init__(self, 
            implicit_layer_dims: List[List[int]],
            vanilla_layer_dims: List[int],
            bias: bool,
            activation = torch.nn.ReLU,
        ):
        super().__init__()
        
        assert bias == False, 'Not implemented'
        assert len(implicit_layer_dims) == len(vanilla_layer_dims)-1, f"len(implicit_layer_dims): {len(implicit_layer_dims)}, len(vanilla_layer_dims): {len(vanilla_layer_dims)}, implicit_layer_dims: {implicit_layer_dims}, vanilla_layer_dims: {vanilla_layer_dims}"

        self.activation = activation()

        self.vanilla_layers = nn.ParameterList()
        self.implicit_layers = nn.ModuleList()
        self.vanilla_layers2 = nn.ModuleList()
        
        for layer_idx, (implicit_layer_dim, vanilla_layer_dim) in enumerate(zip(implicit_layer_dims, vanilla_layer_dims)):
            in_features, out_features = vanilla_layer_dim, vanilla_layer_dims[layer_idx+1]

            if implicit_layer_dim == [0]:
                implicit_layer = nn.Identity() 
            else: 
                n_features = in_features * out_features
                assert n_features == implicit_layer_dim[0],  f"n_features: {n_features}, implicit_layer_dim[0]: {implicit_layer_dim[0]}"
                assert n_features == implicit_layer_dim[-1], f"n_features: {n_features}, implicit_layer_dim[-1]: {implicit_layer_dim[-1]}"

                implicit_layer = nn.Sequential(
                    View((n_features,)),
                    torchvision.ops.MLP(in_channels=implicit_layer_dim[0], hidden_channels=implicit_layer_dim[1:]),
                    View((out_features, in_features)), 
                ) 

            self.implicit_layers.append(implicit_layer)

            vanilla_layer = nn.Linear(in_features, out_features, bias=bias)
            self.vanilla_layers.append(vanilla_layer.weight.data.clone())
            self.vanilla_layers2.append(vanilla_layer)

            # self.phi[f'layer_{layer_idx}_weight'] = weight.weight.data.clone()
            # self.psi[f'layer_{layer_idx}'] = implicit_layer

            # self.vanilla_layers2.append(weight)

            # weight = nn.Parameter(torch.rand(out_features, in_features))
            # nn.init.kaiming_normal_(weight, nonlinearity='relu')
            # self.vanilla_layers.append(weight)

            # self.layers.append(implicit_layer)
            # self.weights.append( nn.Parameter(torch.rand(out_features, in_features)) )

        

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:

        for layer_idx, (implicit_layer, vanilla_layer_weight, vanilla_layer2) in enumerate(zip(self.implicit_layers, self.vanilla_layers, self.vanilla_layers2)):
            
            x_in = x
            # weight_out = implicit_layer(vanilla_layer_weight)
            # x = F.linear(x, weight_out)

            weight_out = implicit_layer(vanilla_layer2.weight)
            x = F.linear(x_in, weight_out) 

            # vanilla_layer2.weight.data = implicit_layer(vanilla_layer2.weight)
            # x2 = vanilla_layer2(x_in)
            # print(torch.allclose(x, x2))

            # weight_sim = torch.allclose(vanilla_layer_weight, vanilla_layer2.weight)
            # print('weight_sim in mlp', weight_sim)

            if layer_idx != len(self.vanilla_layers)-1:
                x = self.activation(x)


        return x


    # def forward(self, x: torch.Tensor) -> List[torch.Tensor]:

    #     outs = [x]

    #     for layer_idx, (layer, weight) in enumerate(zip(self.layers, self.weights)):
    #         weight_out = layer(weight)
    #         self.weights_out.append(weight_out)

    #         x = F.linear(x, weight_out)

            
    #         if layer_idx != len(self.layers)-1:
    #             x = self.activation(x)

    #         outs.append(x)

    #     return x


class EMLP_wrapper(EMLP_MLP):
    def __init__(
            self,
            implicit_layer_dims: List[List[int]],
            vanilla_layer_dims: List[int],
            bias: bool,
            activation = torch.nn.ReLU,
            **kwargs,
        ):
        assert len(implicit_layer_dims) == 0
        ch = vanilla_layer_dims[1:-1]
        repin, repout, group = kwargs['repin'], kwargs['repout'], kwargs['group']
        print(repin, repout, group, ch)
        super().__init__(rep_in=repin, rep_out=repout, group=group, ch=ch, num_layers=None, bias=bias)
    



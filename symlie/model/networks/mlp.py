import torch
import torchvision
from torch import nn
from typing import List, Union
import torch.nn.functional as F
import math
from torch import nn
import os


from emlp.nn.pytorch import MLP as EMLP_MLP
from emlp.nn.pytorch import EMLP

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
    
class ImplicitLayer(nn.Module):
    def __init__(self, implicit_layer_dim, in_features, out_features, forward_type, pretrained = False):
        super().__init__()

        self.n_features = in_features * out_features + out_features # dim of w and b
        self.in_features = in_features
        self.out_features = out_features


        if implicit_layer_dim == [0]:
            self.implicit_layer = nn.Identity() 
        else:
            self.implicit_layer = torchvision.ops.MLP(
                in_channels=implicit_layer_dim[0], 
                hidden_channels=implicit_layer_dim[1:]
            )

        n = 7
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        n = 7
        self.wp1 = torch.zeros((n,)).to(device)
        self.wp2 = self.get_space_translation(n).to(device)


        if pretrained:
            dims_layer0 = [[56, 56, 56, 56], [47, 47, 47, 47], [49, 49, 49, 49], [110, 110, 110, 110]]
            dims_layer1 = [[8, 8, 8], [7, 7, 7], [11, 11, 11, 11]]
            if implicit_layer_dim in dims_layer0:
                print('loading statedict layer0')
                self.implicit_layer.load_state_dict(torch.load('implicit_layer0.pt'))
            if implicit_layer_dim in dims_layer1:
                print('loading statedict layer1')
                self.implicit_layer.load_state_dict(torch.load('implicit_layer1.pt'))

        forward_type_dict = {
            '05synth': self.forward_analytic_05synth,
            'sine1d': self.forward_analytic_sine1d,
            'train': self.forward_train,
        }
        self.forward = forward_type_dict[forward_type]

        # self.implicit_layer = nn.Sequential(
        #         View((self.n_features,)),
        #         torchvision.ops.MLP(in_channels=implicit_layer_dim[0], hidden_channels=implicit_layer_dim[1:]),
        #         View((out_features, in_features+1)), 
        # ) 

    def get_space_translation(self, size):
        w1 = torch.zeros(size)
        w1[0] = 1.
        w2 = torch.stack([torch.roll(w1, shifts = shift) for shift in range(size)])
        w3 = torch.cat([torch.roll(w2, shifts = (shift, 0), dims = (0,1)) for shift in range(size)])

        w_index = torch.zeros(size**2, size**2)
        w_index[:, :size] = w3
        return w_index



    def forward_train(self, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # return w, b
        if type(b) == nn.Parameter or type(b) == torch.Tensor:
            wb = torch.cat([w, b.view(-1, 1)], dim=1).flatten()

            wb = self.implicit_layer(wb)
            wb = wb.view(self.out_features, self.in_features+1)
            w, b = wb[:, :-1], wb[:, -1]

        elif b is None:
            assert b == None
            w = w.flatten()
            w = self.implicit_layer(w)
            w = w.view(self.out_features, self.in_features) 

        return w, b
    
    
    def forward_analytic_05synth(self, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:


        n = 10
        nn = n // 2
        if w.shape == (1, n) and b.shape == (1,):

            wp = torch.cat([torch.zeros(nn), torch.ones(nn)])

            wb = torch.tensor([0])

            w, b = (w.flatten()[wp.flatten().long()]).reshape(w.shape), b.flatten()[wb.long()]
            

        elif w.shape == (n, n) and b.shape == (n,):

            i1 = torch.zeros(nn)
            i1[0] = 1
            i2 = torch.cat([i1])
            i3 = torch.stack([torch.roll(i2, j, 0) for j in range(nn)])
            i4 = i3 + 2
            i5 = torch.cat([i3, i4], dim=1)
            i6 = i5 + 4
            wp = torch.cat([i5, i6], dim=0)

            wb = torch.cat([torch.zeros(nn), torch.ones(nn)])


            w, b = (w.flatten()[wp.flatten().long()]).reshape(w.shape), b.flatten()[wb.long()]

        # if type(b) == nn.Parameter or type(b) == torch.Tensor:
        #     wb = torch.cat([w, b.view(-1, 1)], dim=1).flatten()

        #     wb = self.implicit_layer(wb)
        #     wb = wb.view(self.out_features, self.in_features+1)
        #     w, b = wb[:, :-1], wb[:, -1]

        # elif b is None:
        #     assert b == None
        #     w = w.flatten()
        #     w = self.implicit_layer(w)
        #     w = w.view(self.out_features, self.in_features)

        else:
            raise ValueError(f'Unknown type of b: {b}: {type(b)}')

        return w, b
    
    def forward_analytic_sine1d(self, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:

        n = 7
        if w.shape == (1, n):# and b.shape == (1,):
            w = (w.flatten()[self.wp1.flatten().long()]).reshape(w.shape)
            b = b

            # wp = torch.cat([torch.zeros(nn), torch.ones(nn)])
            # wb = torch.tensor([0])
            # w, b = (w.flatten()[wp.flatten().long()]).reshape(w.shape), b.flatten()[wb.long()]
            
        elif w.shape == (n, n):# and b.shape == (n,):
            w = ( self.wp2 @ w.flatten() ).reshape(w.shape)
            
            if b is not None:
                b = (b.flatten()[self.wp1.flatten().long()]).reshape(b.shape)

        else:
            raise ValueError(f'Unknown type of b: {b}: {type(b)}')

        return w, b



        # if w.shape == (1, 1) and b.shape == (1,):
        #     wp = torch.tensor([0])
        #     wb = torch.tensor([0])
        #     w, b = w.flatten()[wp.long()], b.flatten()[wb.long()]
        # elif w.shape == (2, 1) and b.shape == (2,):
        #     wp = torch.tensor([0, 1])
        #     wb = torch.tensor([0, 1])
        #     w, b = w.flatten()[wp.long()], b.flatten()[wb.long()]
        # else:
        #     raise ValueError(f'Unknown type of b: {b}: {type(b)}')
        # return w, b
    
    # def forward(self, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # return self.forward_analytic_05synth(w, b)
        # return self.forward_analytic_sine1d(w, b)
        # return self.forward_train(w, b)

        
class CombiMLP(torch.nn.Module):
    def __init__(self, 
            implicit_layer_dims: List[List[int]],
            vanilla_layer_dims: List[int],
            bias: bool,
            activation = torch.nn.ReLU,
            pretrained = False,
            forward_type = None,
            # activation = torch.nn.SiLU,
        ):
        super().__init__()
        
        # assert bias == False, 'Not implemented'
        assert forward_type is not None
        assert len(implicit_layer_dims) == len(vanilla_layer_dims)-1, f"len(implicit_layer_dims): {len(implicit_layer_dims)}, len(vanilla_layer_dims): {len(vanilla_layer_dims)}, implicit_layer_dims: {implicit_layer_dims}, vanilla_layer_dims: {vanilla_layer_dims}"

        self.activation = activation()

        self.implicit_layers = nn.ModuleList()
        self.vanilla_layers = nn.ModuleList()
        
        for layer_idx, (implicit_layer_dim, vanilla_layer_dim) in enumerate(zip(implicit_layer_dims, vanilla_layer_dims)):
            in_features, out_features = vanilla_layer_dim, vanilla_layer_dims[layer_idx+1]

            # if implicit_layer_dim == [0]:
            #     implicit_layer = nn.Identity() 
            # else: 
            #     # n_features = in_features * out_features + out_features
            #     # assert n_features == implicit_layer_dim[0]+ out_features,  f"n_features: {n_features}, implicit_layer_dim[0]: {implicit_layer_dim[0]}"
            #     # assert n_features == implicit_layer_dim[-1]+ out_features, f"n_features: {n_features}, implicit_layer_dim[-1]: {implicit_layer_dim[-1]}"
            implicit_layer = ImplicitLayer(implicit_layer_dim, in_features, out_features, forward_type=forward_type, pretrained=pretrained)

            self.implicit_layers.append(implicit_layer)

            vanilla_layer = nn.Linear(in_features, out_features, bias=bias)
            self.vanilla_layers.append(vanilla_layer)

            # self.phi[f'layer_{layer_idx}_weight'] = weight.weight.data.clone()
            # self.psi[f'layer_{layer_idx}'] = implicit_layer

            # self.vanilla_layers2.append(weight)

            # weight = nn.Parameter(torch.rand(out_features, in_features))
            # nn.init.kaiming_normal_(weight, nonlinearity='relu')
            # self.vanilla_layers.append(weight)

            # self.layers.append(implicit_layer)
            # self.weights.append( nn.Parameter(torch.rand(out_features, in_features)) )

        # reset_parameters
        self.reset_parameters_vanilla()



    def reset_parameters_vanilla(self):
        for layer in self.vanilla_layers:
            layer.reset_parameters()

        

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:

        self.x_ins = []

        for layer_idx, (implicit_layer, vanilla_layer) in enumerate(zip(self.implicit_layers, self.vanilla_layers)):
            
            # weight_out = implicit_layer(vanilla_layer_weight)
            # x = F.linear(x, weight_out)
            # bias = vanilla_layer.bias


            with torch.no_grad():
                self.x_ins.append(x)


            weight, bias = implicit_layer(vanilla_layer.weight, vanilla_layer.bias)
            x = F.linear(x, weight, bias)

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


class EMLP_wrapper(EMLP):
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
        # print(implicit_layer_dims, vanilla_layer_dims)
        # print(repin, repout, group, ch)
        super().__init__(rep_in=repin, rep_out=repout, group=group, ch=ch, num_layers=None, bias=bias)
    
class EMLP_MLP_wrapper(EMLP_MLP):
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
    



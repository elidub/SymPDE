from torchvision.transforms import Compose, RandomRotation, CenterCrop, Pad, ToTensor, Resize, RandomCrop
import torch
import numpy as np
from PIL import Image

class UnsqueezeTransform:
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x):
        return torch.unsqueeze(x, dim=self.dim)
    
class SqueezeTransform:
    def __init__(self, dim = None):
        self.dim = dim

    def __call__(self, x):
        if self.dim is None:
            return torch.squeeze(x)
        return torch.squeeze(x, dim=self.dim)
    
class MyPad:
    def __init__(self, pad_width, mode):
        self.pad_width = pad_width
        self.mode = mode
    
    def __call__(self, x):
        return np.pad(x, pad_width = self.pad_width, mode = self.mode)
    
class MyPrint:
    def __init__(self, msg = ''):
        self.msg = msg

    def __call__(self, x):
        print(self.msg, x.shape)
        return x
    
class RandomPad:
    def __init__(self, pad_width):
        self.pad_width = pad_width
    
    def __call__(self, x):
        # print('random pad')
        # pad_widths = np.array([np.random.randint(1, pad) for pad in self.pads])
        # print(pad_width, x.shape)
        # print(pad_widths)
        # assert False
        x = torch.nn.functional.pad(x, pad = (self.pad_width,self.pad_width,self.pad_width,self.pad_width), mode = 'constant', value = 0)
        # print(x.shape)
        return x

    # def __init__(self, pad_width, mode):
    #     self.pad_width = pad_width
    #     self.mode = mode
    
    # def __call__(self, x):
    #     pad_width = np.random.randint(1, self.pad_width)
    #     print(pad_width)
    #     x = x.numpy()
    #     x = np.pad(x, pad_width = pad_width, mode = self.mode)
    #     x = torch.from_numpy(x)
    #     return x
    
class ToArray:
    def __init__(self):
        pass

    def __call__(self, x):
        return x.numpy()
    
class Slice:
    def __init__(self, slice_):
        self.slice_ = slice_

    def __call__(self, x):
        # print('slice')
        # print(x.shape)
        # print(self.slice_)
        x = x[:, :self.slice_, :self.slice_]
        # print(x.shape)
        return x
    
        
class BaseTransform:
    def __init__(self, eps, sample):
        if sample:
            self.eps = torch.rand((1,))*eps
        else:
            self.eps = torch.ones((1,))*eps
        self.eps = self.eps.item()
        
class SpaceTranslate(BaseTransform):
    def __init__(self, dim, eps = 1, sample = True, return_eps = True):
        super().__init__(eps, sample)
        self.dim = dim
        self.return_eps = return_eps

    def __repr__(self) -> str:
        return f'SpaceTranslate_{self.dim}'

    def __call__(self, x):
        grid_size = x.shape[self.dim]
        shift = int(grid_size*self.eps)
        x = torch.roll(x, shifts = shift, dims = self.dim)
        if self.return_eps:
            return x, self.eps
        else:
            return x
    
class RandomScale(BaseTransform):
    def __init__(self, eps = 1, sample = True):
        super().__init__(eps, sample)
        self.mult = 1 + self.eps - 0.5

    def __call__(self, x):
        return x*self.mult

    # def __call__(self, x):
    #     _, h, w = x.shape
    #     assert h == w
    #     pads = tuple([int(w//2*self.eps) for _ in range(4)])
    #     x = torch.nn.functional.pad(x, pad = pads, mode = 'constant', value = 0)
    #     return x
    
class CustomRandomRotation(BaseTransform):
    def __init__(self, eps = 1, interpolation=Image.BILINEAR, sample = True):
        super().__init__(eps, sample)

        r = self.eps*180
        self.rotate = RandomRotation(degrees = (-r, r), fill = 0., interpolation=interpolation)

    def __str__(self) -> str:
        return super().__str__()

    def __call__(self, x):
        # print(x.shape)
        x = self.rotate(x)
        # x = torch.nn.functional.rotate(x, angle = angle, mode = 'Image.BILINEAR')
        return x
    
class CustomCompose:
    def __init__(self, transforms, augment_kwargs = {}):
        self.transforms = transforms
        self.augment_kwargs = augment_kwargs

    def recenter(self, x, augment_kwargs):
        transform = Compose([
            # Space translation
            SpaceTranslate(eps=-augment_kwargs['SpaceTranslate_1'], dim = 1, sample=False, return_eps=False), # x translation
            SpaceTranslate(eps=-augment_kwargs['SpaceTranslate_2'], dim = 2, sample=False, return_eps=False), # y translation
        ])
        return transform(x)


    def __call__(self, x):
        augment_kwargs_out = {}
        x = self.recenter(x, self.augment_kwargs)
        for transform, in zip(self.transforms,):
            transform_key = transform.__repr__()
            out = transform(x)
            if isinstance(out, tuple):
                x, augment_kwarg_out = out
                assert transform_key not in augment_kwargs_out
                augment_kwargs_out[transform_key] = augment_kwarg_out
            else:
                x = out
        return x, augment_kwargs_out
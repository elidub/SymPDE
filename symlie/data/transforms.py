from torchvision.transforms import Compose, RandomRotation, CenterCrop, Pad, ToTensor, Resize, RandomCrop
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import RandomAffine, InterpolationMode
from tqdm import tqdm

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
        raise NotImplementedError
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
        
        
class SpaceTranslate():
    def __init__(self, dim, return_shift = True):
        self.dim = dim
        self.return_shift = return_shift

    def __repr__(self) -> str:
        return f'SpaceTranslate_{self.dim}'

    def __call__(self, x, eps):
        grid_size = x.shape[self.dim]
        shift = int(grid_size*eps)
        x = torch.roll(x, shifts = shift, dims = self.dim)
        if self.return_shift:
            shifts = torch.full(size=(len(x),), fill_value=shift)
            return x, shifts
        else:
            return x
    
class RandomScale():
    def __init__(self, l = 2):
        self.l = l

    def __call__(self, x, eps):
        mult = self.l ** eps
        return x*mult

class CustomRandomRotation():
    def __init__(self, only_flip: bool):
        self.only_flip = only_flip
        
    def __str__(self) -> str:
        return super().__str__()

    def __call__(self, x, eps):
        k = int(eps*4)
        
        if self.only_flip:
            k=k*2 # make it even to prevent 90deg rotation

        x = torch.rot90(x, k=k, dims=(1,2))
        return x
    
    

class Transform:
    def __init__(self, only_flip: bool = False):
        self.scale = RandomScale()

        self.rotate = CustomRandomRotation(only_flip = only_flip)

        self.space_translate_x = SpaceTranslate(dim = 2)
        self.space_translate_y = SpaceTranslate(dim = 1)

    def batch_space_translate(self, x: torch.Tensor, shifts: torch.Tensor, shift_dir: str) -> torch.Tensor:
        """Translate a batch of images by a specified number of pixels.

        Args:
            x: Batch of images of shape (batch_size, height, width).
            shifts: Batch of shifts of shape (batch_size, 2).

        Returns:
            Batch of translated images of shape (batch_size, height, width).
        """
        batch_size, height, width = x.shape
        assert height == width

    
        # Create an index tensor
        indices = torch.arange(height).unsqueeze(0).expand(len(shifts), -1)

        # Use broadcasting to create tensor b
        b = indices < shifts.unsqueeze(1)
        if shift_dir == 'x': x = torch.transpose(x, 1, 2)
        assert (x == 0.).sum() == 0, x
        # i1, i2 = 1, torch.nan
        i1, i2 = torch.tensor(1.), torch.tensor(0.)
        b21 = torch.where(b, i1, i2).unsqueeze(2)
        b22 = torch.where(b, i2, i1).unsqueeze(2)
        x = torch.cat([x * b22, x * b21], dim = 1)
        # indices2 = torch.where(~torch.isnan(x))
        indices2 = torch.where(x != 0.)
        x = x[[*indices2]].view(batch_size, height, width)
        if shift_dir == 'x': x = torch.transpose(x, 1, 2)

        return x

    def recenter(self, x, centers):
        centers_x, centers_y = centers.T
        x = self.batch_space_translate(x, centers_x, shift_dir = 'x')
        x = self.batch_space_translate(x, centers_y, shift_dir = 'y')
        return x
    
    def transform(self, x, centers, epsilons):
        batch_size, features = x.shape

        grid_size = int(np.sqrt(features))
        x = x.reshape(batch_size, grid_size, grid_size)
        # x = x.unsqueeze(1)

        # x = self.recenter(x, centers)

        # x = self.scale(x, epsilons[0])

        # x = self.rotate(x, epsilons[1])


        x, centers_x = self.space_translate_x(x, epsilons[2])
        x, centers_y = self.space_translate_y(x, epsilons[3])

        x = x.reshape(batch_size, features)

        centers_new = torch.stack([centers_x, centers_y], dim = 1)
        return x, centers_new

    def transform_individual(self, x, centers, epsilons):
        xs, centers = zip(*[self.transform(x_i, centers_i, epsilons_i) for x_i, centers_i, epsilons_i in tqdm(zip(x.unsqueeze(1), centers.unsqueeze(1), epsilons), total = len(x), leave=False, disable = True)])
        x, centers = torch.cat(xs), torch.cat(centers)
        return x, centers

    def __call__(self, x, centers, epsilons, transform_individual_bool = False):
        if transform_individual_bool:
            return self.transform_individual(x, centers, epsilons)
        else:
            return self.transform(x, centers, epsilons)



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
    def __init__(self, sample):
        self.sample = sample

    def get_eps(self, eps):
        if self.sample:
            eps = torch.rand((1,))*eps
        else:
            eps = torch.ones((1,))*eps
        return eps.item()

        
class SpaceTranslate(BaseTransform):
    def __init__(self, dim, max_eps = 1, sample = True, return_shift = True):
        super().__init__(sample)
        self.max_eps = max_eps
        self.dim = dim
        self.return_shift = return_shift

    def __repr__(self) -> str:
        return f'SpaceTranslate_{self.dim}'

    def __call__(self, x):
        eps = self.get_eps(self.max_eps)
        grid_size = x.shape[self.dim]
        shift = int(grid_size*eps)
        x = torch.roll(x, shifts = shift, dims = self.dim)
        if self.return_shift:
            shifts = torch.full(size=(len(x),), fill_value=shift)
            return x, shifts
        else:
            return x
    
class RandomScale(BaseTransform):
    def __init__(self, max_eps = 1, sample = True):
        super().__init__(sample)
        # self.mult = 1 + self.eps - 0.5

    def __call__(self, x):
        # return x*self.mult
        return x

    # def __call__(self, x):
    #     _, h, w = x.shape
    #     assert h == w
    #     pads = tuple([int(w//2*self.eps) for _ in range(4)])
    #     x = torch.nn.functional.pad(x, pad = pads, mode = 'constant', value = 0)
    #     return x
    
class CustomRandomRotation(BaseTransform):
    def __init__(self, max_eps = 1, sample = True):
        super().__init__(sample)
        self.max_eps = max_eps

        # if sample:
        #     r = eps*180
        #     self.rotate = RandomRotation(degrees = (-r, r), fill = 0., interpolation=interpolation)
        # else:
        #     r = eps*360
        #     self.rotate = RandomRotation(degrees = (r, r), fill = 0., interpolation=interpolation)



    def __str__(self) -> str:
        return super().__str__()

    def __call__(self, x):
        eps = self.get_eps(self.max_eps)
        k = int(eps*4)
        x = torch.rot90(x, k=k, dims=(1,2))
        # x = self.rotate(x)
        # x = torch.nn.functional.rotate(x, angle = angle, mode = 'Image.BILINEAR')
        return x
    
    

class Transform:
    def __init__(self, epsilons = [1., 1., 1., 1.], sample = True, antialias= False):
        self.scale = RandomScale(max_eps=epsilons[0], sample=sample)

        self.rotate = Compose([
            CustomRandomRotation(max_eps=epsilons[1], sample=sample),
        ])

        self.space_translate_x = SpaceTranslate(max_eps=epsilons[2], dim = 1, sample=sample)
        self.space_translate_y = SpaceTranslate(max_eps=epsilons[3], dim = 2, sample=sample)

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
        b2 = torch.where(b, torch.tensor(1), torch.tensor(0)).unsqueeze(2)

        if shift_dir == 'x': x = torch.transpose(x, 1, 2)
        i1, i2 = 1, torch.nan
        b21 = torch.where(b, i1, i2).unsqueeze(2)
        b22 = torch.where(b, i2, i1).unsqueeze(2)
        x = torch.cat([x * b22, x * b21], dim = 1)
        indices2 = torch.where(~torch.isnan(x))
        x = x[[*indices2]].view(batch_size, height, width)
        if shift_dir == 'x': x = torch.transpose(x, 1, 2)

        return x

    def recenter(self, x, centers):
        centers_x, centers_y = centers.T
        x = self.batch_space_translate(x, centers_x, shift_dir = 'x')
        x = self.batch_space_translate(x, centers_y, shift_dir = 'y')
        return x
    
    
    def transform(self, x, centers):
        x = self.recenter(x, centers)

        x = self.scale(x)

        x = self.rotate(x)

        x, centers_x = self.space_translate_x(x)
        x, centers_y = self.space_translate_y(x)

        centers_new = torch.stack([centers_x, centers_y], dim = 1)
        return x, centers_new
    
    def transform_individual(self, x, centers):
        xs, centers = zip(*[self.transform(x_i, centers=centers_i) for x_i, centers_i in tqdm(zip(x.unsqueeze(1), centers.unsqueeze(1)), total = len(x), leave=False, disable = True)])
        x, centers = torch.cat(xs), torch.cat(centers)
        return x, centers

    def __call__(self, x, centers, transform_individual_bool = False):
        if transform_individual_bool:
            return self.transform_individual(x, centers)
        else:
            return self.transform(x, centers)




# class CustomCompose:
#     def __init__(self, transforms, centers):
#         self.transforms = transforms
#         self.centers = centers

#     def recenter(self, x, augment_kwargs):
#         transform = Compose([
#             # Space translation
#             SpaceTranslate(eps=-augment_kwargs['SpaceTranslate_1'], dim = 1, sample=False, return_eps=False), # x translation
#             SpaceTranslate(eps=-augment_kwargs['SpaceTranslate_2'], dim = 2, sample=False, return_eps=False), # y translation
#         ])
#         return transform(x)


#     def __call__(self, x):
#         augment_kwargs_out = {}
#         x = self.recenter(x, self.centers)
#         for transform, in zip(self.transforms,):
#             transform_key = transform.__repr__()
#             out = transform(x)
#             if isinstance(out, tuple):
#                 x, augment_kwarg_out = out
#                 assert transform_key not in augment_kwargs_out
#                 augment_kwargs_out[transform_key] = augment_kwarg_out
#             else:
#                 x = out
#         return x, augment_kwargs_out
    
# class CustomCompose:
#     def __init__(self, epsilons, sample, antialias):
#         pass

#     def recenter(self, x, augment_kwargs):
#         transform = Compose([
#             # Space translation
#             SpaceTranslate(eps=-augment_kwargs['SpaceTranslate_1'], dim = 1, sample=False, return_eps=False), # x translation
#             SpaceTranslate(eps=-augment_kwargs['SpaceTranslate_2'], dim = 2, sample=False, return_eps=False), # y translation
#         ])
#         return transform(x)


#     def __call__(self, x):
#         augment_kwargs_out = {}
#         x = self.recenter(x, self.centers)
#         for transform, in zip(self.transforms,):
#             transform_key = transform.__repr__()
#             out = transform(x)
#             if isinstance(out, tuple):
#                 x, augment_kwarg_out = out
#                 assert transform_key not in augment_kwargs_out
#                 augment_kwargs_out[transform_key] = augment_kwarg_out
#             else:
#                 x = out
#         return x, augment_kwargs_out
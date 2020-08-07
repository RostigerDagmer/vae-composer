import h5py
import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

def view_pianoroll(batch, url=None):
    fig = plt.subplot()
    fig.set_title(url)
    img_grid = torchvision.utils.make_grid(batch, nrow=1)
    for i in range(img_grid.shape[1]):
        if i % 96 == 0:
            img_grid[:,i,:] = torch.ones(img_grid.shape[2]) * 4.0
    fig.imshow(img_grid.transpose(0, 2), url=url)
    plt.show()

def get_pianoroll(batch, name=None):
    img_grid = torchvision.utils.make_grid(batch, nrow=1)
    return Image.fromarray(img_grid.numpy())

def log_volume(t, r=None):
    print(f'tensor shape:{t.shape}')
    vol = np.product(t.shape[1:])
    r_vol = vol if not r else r
    print(f'absolute tensor volume:{vol}')
    print(f'volume reduction:{vol / r_vol}')
    #print(f'latent space development:{str(chr(9608)) * int(20. / (vol / r_vol))}')
    return vol

def tensor_memory(a, return_val=False):
    if return_val:
        return convert_size(a.element_size() * a.nelement())
    else:
        size, size_name = convert_size(a.element_size() * a.nelement())
        print(f'memory {size, " ", size_name}')

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return s, size_name[i]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

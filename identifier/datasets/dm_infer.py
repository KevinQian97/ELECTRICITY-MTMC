from __future__ import absolute_import
from __future__ import print_function

from torch.utils.data import DataLoader

from . import init_imgreid_dataset
from utils.torch_func import  get_mean_and_std, calculate_mean_and_std
from PIL import Image
import os.path as osp
from torch.utils.data import Dataset
from torchvision.transforms import *
from collections import defaultdict
import numpy as np
import copy
import random

from torch.utils.data.sampler import Sampler, RandomSampler



def build_transforms(height,
                     width,
                     **kwargs):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = Normalize(mean=imagenet_mean, std=imagenet_std)

    transform_test = Compose([
        Resize((height, width)),
        ToTensor(),
        normalize,
    ])

    return transform_test

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError('{} does not exist'.format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path

class BaseDataManager(object):

    def __init__(self,
                 use_gpu,
                 test_set,
                 root='imgs',
                 height=128,
                 width=256,
                 test_batch_size=100,
                 workers=4,
                 num_instances=4,  # number of instances per identity (for RandomIdentitySampler)
                 **kwargs
                 ):
        self.use_gpu = use_gpu
        self.test_set = test_set
        self.root = root
        self.height = height
        self.width = width
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.num_instances = num_instances

        transform_test = build_transforms(self.height, self.width)
        self.transform_test = transform_test

    def return_dataloaders(self):
        """
        Return trainloader and testloader dictionary
        """
        return self.testloader_dict



class ImageDataManager(BaseDataManager):
    def __init__(self,
                 use_gpu,
                 test_set,
                 **kwargs
                 ):
        super(ImageDataManager, self).__init__(use_gpu, test_set, **kwargs)

        print('=> Initializing TEST datasets')
        self.testdataset_dict = {"query":None,"test":None}
        self.testloader_dict = {"query":None,"test":None}

        dataset = init_imgreid_dataset(
        root=self.root, name=test_set)

        self.testloader_dict['query'] = DataLoader(
                ImageDataset(dataset.query, transform=self.transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=False
            )

        self.testloader_dict['test'] = DataLoader(
                ImageDataset(dataset.test, transform=self.transform_test),
                batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
                pin_memory=self.use_gpu, drop_last=False
            )

        self.testdataset_dict['query'] = dataset.query
        self.testdataset_dict['test'] = dataset.test


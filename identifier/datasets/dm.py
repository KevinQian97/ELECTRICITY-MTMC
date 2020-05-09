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


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


def build_train_sampler(data_source,
                        train_sampler,
                        train_batch_size,
                        num_instances,
                        **kwargs):
    """Build sampler for training
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - train_sampler (str): sampler name (default: RandomSampler).
    - train_batch_size (int): batch size during training.
    - num_instances (int): number of instances per identity in a batch (for RandomIdentitySampler).
    """

    if train_sampler == 'RandomIdentitySampler':
        sampler = RandomIdentitySampler(data_source, train_batch_size, num_instances)

    else:
        sampler = RandomSampler(data_source)

    return sampler

class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.
    Args:
    - height (int): target image height.
    - width (int): target image width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    Origin: https://github.com/zhunzhong07/Random-Erasing
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class ColorAugmentation(object):
    """
    Randomly alter the intensities of RGB channels
    Reference:
    Krizhevsky et al. ImageNet Classification with Deep ConvolutionalNeural Networks. NIPS 2012.
    """

    def __init__(self, p=0.5):
        self.p = p
        self.eig_vec = torch.Tensor([
            [0.4009, 0.7192, -0.5675],
            [-0.8140, -0.0045, -0.5808],
            [0.4203, -0.6948, -0.5836],
        ])
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def _check_input(self, tensor):
        assert tensor.dim() == 3 and tensor.size(0) == 3

    def __call__(self, tensor):
        if random.uniform(0, 1) > self.p:
            return tensor
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor


def build_transforms(height,
                     width,
                     random_erase=False,  # use random erasing for data augmentation
                     color_jitter=False,  # randomly change the brightness, contrast and saturation
                     color_aug=False,  # randomly alter the intensities of RGB channels
                     **kwargs):
    # use imagenet mean and std as default
    # TODO: compute dataset-specific mean and std
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = Normalize(mean=imagenet_mean, std=imagenet_std)

    # build train transformations
    transform_train = []
    transform_train += [Random2DTranslation(height, width)]
    transform_train += [RandomHorizontalFlip()]
    if color_jitter:
        transform_train += [ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0)]
    transform_train += [ToTensor()]
    if color_aug:
        transform_train += [ColorAugmentation()]
    transform_train += [normalize]
    if random_erase:
        transform_train += [RandomErasing()]
    transform_train = Compose(transform_train)

    # build test transformations
    transform_test = Compose([
        Resize((height, width)),
        ToTensor(),
        normalize,
    ])

    return transform_train, transform_test

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
                 train_sets,
                 test_set,
                 root='imgs',
                 height=128,
                 width=256,
                 train_batch_size=32,
                 test_batch_size=100,
                 workers=4,
                 train_sampler='',
                 random_erase=False,  # use random erasing for data augmentation
                 color_jitter=False,  # randomly change the brightness, contrast and saturation
                 color_aug=False,  # randomly alter the intensities of RGB channels
                 num_instances=4,  # number of instances per identity (for RandomIdentitySampler)
                 **kwargs
                 ):
        self.use_gpu = use_gpu
        self.train_sets = train_sets
        self.test_set = test_set
        self.root = root
        self.height = height
        self.width = width
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.train_sampler = train_sampler
        self.random_erase = random_erase
        self.color_jitter = color_jitter
        self.color_aug = color_aug
        self.num_instances = num_instances

        transform_train, transform_test = build_transforms(
            self.height, self.width, random_erase=self.random_erase, color_jitter=self.color_jitter,
            color_aug=self.color_aug
        )
        self.transform_train = transform_train
        self.transform_test = transform_test

    @property
    def num_train_pids(self):
        return self._num_train_pids

    @property
    def num_train_cams(self):
        return self._num_train_cams

    def return_dataloaders(self):
        """
        Return trainloader and testloader dictionary
        """
        return self.trainloader, self.testloader_dict



class ImageDataManager(BaseDataManager):
    """
    Vehicle-ReID data manager
    """
    def __init__(self,
                 use_gpu,
                 train_sets,
                 test_set,
                 **kwargs
                 ):
        super(ImageDataManager, self).__init__(use_gpu, train_sets, test_set, **kwargs)

        print('=> Initializing TRAIN (source) datasets')
        train = []
        self._num_train_pids = 0
        self._num_train_cams = 0

        for name in self.train_sets:
            dataset = init_imgreid_dataset(
                root=self.root, name=name)

            for img_path, pid, camid in dataset.train:
                pid += self._num_train_pids
                camid += self._num_train_cams
                train.append((img_path, pid, camid))

            self._num_train_pids += dataset.num_train_pids
            self._num_train_cams += dataset.num_train_cams

        self.train_sampler = build_train_sampler(
            train, self.train_sampler,
            train_batch_size=self.train_batch_size,
            num_instances=self.num_instances,
        )
        self.trainloader = DataLoader(
            ImageDataset(train, transform=self.transform_train), sampler=self.train_sampler,
            batch_size=self.train_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.use_gpu, drop_last=True
        )
        mean, std = calculate_mean_and_std(self.trainloader, len(train))
        print('mean and std:', mean, std)

        print('=> Initializing TEST datasets')
        self.testloader_dict = {'query': None, 'test': None}
        self.testdataset_dict = {'query': None, 'test': None} 

        dataset = init_imgreid_dataset(
            root=self.root, name=self.test_set)

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

        print('\n')
        print('  **************** Summary ****************')
        print('  train names      : {}'.format(self.train_sets))
        print('  # train datasets : {}'.format(len(self.train_sets)))
        print('  # train ids      : {}'.format(self.num_train_pids))
        print('  # train images   : {}'.format(len(train)))
        print('  # train cameras  : {}'.format(self.num_train_cams))
        print('  test names       : {}'.format(self.test_set))
        print('  *****************************************')
        print('\n')

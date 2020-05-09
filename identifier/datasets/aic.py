from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import re
import os.path as osp

from .base import BaseImageDataset


class Aic(BaseImageDataset):
    """
    Aic
    """
    dataset_dir = 'Aic'

    def __init__(self, root='imgs', verbose=False, **kwargs):
        super(Aic, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.test_dir = osp.join(self.dataset_dir, 'image_test')

        self.check_before_run()

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        test = self.process_dir(self.test_dir, relabel=False)

        if verbose:
            print('=> Aic loaded')
            self.print_dataset_statistics(train, query, test)

        self.train = train
        self.query = query
        self.test = test

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_test_pids, self.num_test_imgs, self.num_test_cams = self.get_imagedata_info(self.test)

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError('"{}" is not available'.format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError('"{}" is not available'.format(self.query_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError('"{}" is not available'.format(self.test_dir))

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue 
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            # print(pid,camid)
            if pid == -1:
                continue 
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

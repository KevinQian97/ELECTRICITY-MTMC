from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import re
import os.path as osp

from .base import BaseImageDataset


class aic_test(BaseImageDataset):
    """
    inference folder
    """
    dataset_dir = 'aic_test'

    def __init__(self, root='exp/imgs', **kwargs):
        super(aic_test, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.test_dir = osp.join(self.dataset_dir, 'image_test')

        query = self.process_dir(self.query_dir, relabel=False)
        test = self.process_dir(self.test_dir, relabel=False)


        self.query = query
        self.test = test

        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_test_pids, self.num_test_imgs, self.num_test_cams = self.get_imagedata_info(self.test)


    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            # print(pid,camid)
            if pid == -1:
                continue  # junk images are just ignored
            # assert 0 <= pid <= 1501  # pid == 0 means background
            # assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

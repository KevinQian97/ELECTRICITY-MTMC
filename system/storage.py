import os.path as osp
import pickle


class Saver(object):

    def __init__(self, file_name, parent_dir=''):
        self.path = osp.join(parent_dir, file_name + '.pk')

    def save(self, generator):
        with open(self.path, 'wb') as f:
            for obj in generator:
                pickle.dump(obj, f)
                yield obj


class Loader(object):

    def __init__(self, file_name, parent_dir=''):
        self.path = osp.join(parent_dir, file_name + '.pk')

    def load(self):
        with open(self.path, 'rb') as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from opts import argument_parser, testset_kwargs
from datasets.dm_infer import ImageDataManager
import models
from losses import CrossEntropyLoss, TripletLoss, DeepSupervision
from utils.io import check_isfile
from utils.avgmeter import AverageMeter
from utils.log import Logger, RankLogger
from utils.torch_func import count_num_param,load_pretrained_weights
from utils.seed import set_random_seed
from postprocess.postprocess import calc_reid,update_output

# global variables
parser = argument_parser()
args = parser.parse_args()


def main():
    global args
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    print('Initializing image data manager')
    dm = ImageDataManager(use_gpu, **testset_kwargs(args))
    testloader_dict = dm.return_dataloaders()

    print('Initializing model: {}'.format(args.arch))
    model = models.init_model(name=args.arch, loss={'xent', 'htri'},
                              pretrained=not args.no_pretrained, use_gpu=use_gpu)
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    model = nn.DataParallel(model).cuda() if use_gpu else model

    print('Matching {} ...'.format(args.test_set))
    queryloader = testloader_dict['query']
    galleryloader = testloader_dict['test']
    distmat, q_pids, g_pids, q_camids, g_camids = run(model, queryloader, galleryloader, use_gpu, return_distmat=True)
    np.savez(args.save_npy,distmat=distmat,q_pids=q_pids,g_pids=g_pids,q_camids=q_camids,g_camids=g_camids)
    result  = np.load(args.save_npy)
    reid_dict,rm_dict = calc_reid(result)
    print(rm_dict,reid_dict)
    with open(args.tracklet_path,"r") as f:
        or_tracks = f.readlines()
    g = open(args.track3_path,"w")
    update_output(or_tracks,reid_dict,rm_dict,g)


  




def run(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, args.test_batch_size))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    return distmat, q_pids, g_pids, q_camids, g_camids


if __name__ == '__main__':
    main()


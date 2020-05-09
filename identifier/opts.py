import argparse


def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('--root', type=str, default='./exp/imgs',
                        help='root path to data directory')
    parser.add_argument('--train_sets', type=str, nargs='+',
                        help='training datasets for training(delimited by space)')
    parser.add_argument('--test_set', type=str,
                        help='test dataset for testing or validation')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (tips: 4 or 8 times number of gpus)')
    parser.add_argument('--height', type=int, default=128,
                        help='height of an image')
    parser.add_argument('--width', type=int, default=256,
                        help='width of an image')
    parser.add_argument('--train-sampler', type=str, default='RandomSampler',
                        help='sampler for trainloader')
    parser.add_argument('--save_npy', type=str, default='./exp/distmatrix.npz')
    parser.add_argument("--tracklet_path",type =str, default= "./exp/tracklets.txt")
    parser.add_argument("--track3_path",type =str, default="./exp/track3.txt")
    # ************************************************************
    # Data augmentation(not used in submission system)
    # ************************************************************
    parser.add_argument('--random-erase', action='store_true',
                        help='use random erasing for data augmentation')
    parser.add_argument('--color-jitter', action='store_true',
                        help='randomly change the brightness, contrast and saturation')
    parser.add_argument('--color-aug', action='store_true',
                        help='randomly alter the intensities of RGB channels')
    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='amsgrad',
                        help='optimization algorithm')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help='weight decay')
    # sgd
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum factor for sgd and rmsprop')
    parser.add_argument('--sgd-dampening', default=0, type=float,
                        help='sgd\'s dampening for momentum')
    parser.add_argument('--sgd-nesterov', action='store_true',
                        help='whether to enable sgd\'s Nesterov momentum')
    # rmsprop
    parser.add_argument('--rmsprop-alpha', default=0.99, type=float,
                        help='rmsprop\'s smoothing constant')
    # adam/amsgrad
    parser.add_argument('--adam-beta1', default=0.9, type=float,
                        help='exponential decay rate for adam\'s first moment')
    parser.add_argument('--adam-beta2', default=0.999, type=float,
                        help='exponential decay rate for adam\'s second moment')

    # ************************************************************
    # Training hyperparameters
    # ************************************************************
    parser.add_argument('--max-epoch', default=25, type=int,
                        help='maximum epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful when restart)')

    parser.add_argument('--train-batch-size', default=128, type=int,
                        help='training batch size')
    parser.add_argument('--test-batch-size', default=256, type=int,
                        help='test batch size')

    # ************************************************************
    # Learning rate scheduler options
    # ************************************************************
    parser.add_argument('--lr-scheduler', type=str, default='multi_step',
                        help='learning rate scheduler (see lr_schedulers.py)')
    parser.add_argument('--stepsize', default=[10, 15, 20], nargs='+', type=int,
                        help='stepsize to decay learning rate')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='learning rate decay')

    # ************************************************************
    # Cross entropy loss-specific setting
    # ************************************************************
    parser.add_argument('--label-smooth', action='store_true',
                        help='use label smoothing regularizer in cross entropy loss')

    # ************************************************************
    # Hard triplet loss-specific setting
    # ************************************************************
    parser.add_argument('--margin', type=float, default=0.3,
                        help='margin for triplet loss')
    parser.add_argument('--num-instances', type=int, default=4,
                        help='number of instances per identity')
    parser.add_argument('--lambda-xent', type=float, default=1,
                        help='weight to balance cross entropy loss')
    parser.add_argument('--lambda-htri', type=float, default=1,
                        help='weight to balance hard triplet loss')

    # ************************************************************
    # Architecture
    # ************************************************************
    parser.add_argument('-a', '--arch', type=str, default='resnet101')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='do not load pretrained weights')

    # ************************************************************
    # Test settings
    # ************************************************************
    parser.add_argument('--load-weights', type=str, default='',
                        help='load pretrained weights but ignore layers that do not match in size')
    parser.add_argument('--eval-freq', type=int, default=1,
                        help='evaluation frequency (set to -1 to test only in the end)')
    parser.add_argument('--start-eval', type=int, default=0,
                        help='start to evaluate after a specific epoch')
    parser.add_argument('--query-remove', type=bool, default=True)
    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--print-freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--seed', type=int, default=1,
                        help='manual seed')
    parser.add_argument('--resume', type=str, default='', metavar='PATH',
                        help='resume from a checkpoint')
    parser.add_argument('--save-dir', type=str, default='log',
                        help='path to save log and model weights')
    parser.add_argument('--use-cpu', action='store_true',
                        help='use cpu')
    parser.add_argument('--gpu-devices', default='0', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--use-avai-gpus', action='store_true',
                        help='use all available gpus instead of specified devices')
    return parser


def trainset_kwargs(parsed_args):
    """
    Build kwargs for ImageDataManager in dm.py and dm_infer.py from
    the parsed command-line arguments.
    """
    return {
        'train_sets': parsed_args.train_sets,
        'test_set': parsed_args.test_set,
        'root': parsed_args.root,
        'height': parsed_args.height,
        'width': parsed_args.width,
        'train_batch_size': parsed_args.train_batch_size,
        'test_batch_size': parsed_args.test_batch_size,
        'workers': parsed_args.workers,
        'train_sampler': parsed_args.train_sampler,
        'random_erase': parsed_args.random_erase,
        'color_jitter': parsed_args.color_jitter,
        'color_aug': parsed_args.color_aug,
    }

def testset_kwargs(parsed_args):
    """
    Build kwargs for ImageDataManager in dm.py and dm_infer.py from
    the parsed command-line arguments.
    """
    return {
        'test_set': parsed_args.test_set,
        'root': parsed_args.root,
        'height': parsed_args.height,
        'width': parsed_args.width,
        'test_batch_size': parsed_args.test_batch_size,
        'workers': parsed_args.workers,
    }


def optimizer_kwargs(parsed_args):
    """
    Build kwargs for optimizer in torch_func.py from
    the parsed command-line arguments.
    """
    return {
        'optim': parsed_args.optim,
        'lr': parsed_args.lr,
        'weight_decay': parsed_args.weight_decay,
        'momentum': parsed_args.momentum,
        'sgd_dampening': parsed_args.sgd_dampening,
        'sgd_nesterov': parsed_args.sgd_nesterov,
        'rmsprop_alpha': parsed_args.rmsprop_alpha,
        'adam_beta1': parsed_args.adam_beta1,
        'adam_beta2': parsed_args.adam_beta2,
    }


def lr_scheduler_kwargs(parsed_args):
    """
    Build kwargs for lr_scheduler in torch_func.py from
    the parsed command-line arguments.
    """
    return {
        'lr_scheduler': parsed_args.lr_scheduler,
        'stepsize': parsed_args.stepsize,
        'gamma': parsed_args.gamma,
    }

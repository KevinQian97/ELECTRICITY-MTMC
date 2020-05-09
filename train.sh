set -e
dir=$(dirname $(readlink -fn --  $0))
python ./identifier/train.py --train_sets Aic --test_set Aic --train-batch-size 128 \
 --test-batch-size 256 -a resnet101 --save-dir models/resnet101-Aic --use-avai-gpus \
 --root ./datasets

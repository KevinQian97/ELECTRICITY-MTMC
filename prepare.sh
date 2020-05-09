set -e
dir=$(dirname $(readlink -fn --  $0))
python ./identifier/preprocess/prepare_dataset.py 
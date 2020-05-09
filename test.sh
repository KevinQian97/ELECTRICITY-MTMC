#please download the data set and locate it under ./datasets
set -e
dir=$(dirname $(readlink -fn --  $0))
cd ../
python -m ELECTRICITY-MTMC.utils.test

cd $dir
python ./identifier/preprocess/extract_img.py

python ./identifier/test.py --test-batch-size 256 --test_set aic_test \
 --use-avai-gpus --load-weights ./models/resnet101-Aic/model.pth.tar-9 

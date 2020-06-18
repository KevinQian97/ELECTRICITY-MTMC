# ELECTRICITY: An Efficient Multi-camera Vehicle Tracking System for Intelligent City

Authors: Yijun Qian, Lijun Yu, Wenhe Liu, Alexander G Hauptmann

Email: yijunqia@andrew.cmu.edu, lijun@cmu.edu
[Paper](http://openaccess.thecvf.com/content_CVPRW_2020/papers/w35/Qian_ELECTRICITY_An_Efficient_Multi-Camera_Vehicle_Tracking_System_for_Intelligent_City_CVPRW_2020_paper.pdf)
```bib
@inproceedings{qian2020electricity,
  title={ELECTRICITY: An Efficient Multi-camera Vehicle Tracking System for Intelligent City},
  author={Qian, Yijun and Yu, Lijun and Liu, Wenhe and Hauptmann, Alexander G.},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2020}
}
```

## Overview
We release the code for our winnning model on AI City 2020 Challenge (https://www.aicitychallenge.org/) Track 3.
For more information please refer to our accepted paper in CVPR 2020 AI City Workshop.

## Project Download

Firstly please download the project through:
```
git clone https://github.com/KevinQian97/ELECTRICITY-MTMC.git
```

## Prerequisites
The code is built with many libraries, we have listed the official sites of part of them.
If you encounter problems about the dependencies, please resort to these official sites for help.
- [PyTorch](https://pytorch.org/) 1.5 (we did not try other versions of pytorch)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [Opencv](https://opencv.org/)
- [Cython](https://cython.org/)
- [AV](https://github.com/PyAV-Org/PyAV)
- [ANACONDA](https://www.anaconda.com/)

We have prepared the environment config file and suggest build the environment through ANACONDA.
```
cd ELECTRICITY-MTMC
conda env create -f environment.yml 
conda activate aic20_track3
```

## Data Preparation
If you want to reproduce our results on AI City Challenge or train the model by yourself, please download the data set from: (https://www.aicitychallenge.org/2020-data-and-evaluation/)
and put it under the folder datasets.
Make sure the data structure is like:
* ELECTRICITY-MTMC
  * datasets
    * aic_20_trac3
      * test (test folder)
      * eval 
      * validation (validation folder)
      * cam_timestamp
      * cam_loc
      * cam_framenum
      * train (train folder) 


## Pretrained Models
We also provided the pretrained model:
Notice:The accuracy and map here is calculated on our inner split of validation set.
| model             | Acc 1 |  MAP  | Epochs | Linkage                                                                                   |
| ----------------- | ----- | ----- | ------ | ----------------------------------------------------------------------------------------- |
| Agg_ResNet101     | 92.0% | 82.3% |   10   | [link](https://drive.google.com/file/d/1Z6E0h2qh3QWnfcj3kmt5UPXUz9EdSN0-/view?usp=sharing)|
## Inference
If you just want inference or reproduce our results, you can directly download our pretrained model and:
```
cd ELECTRICITY-MTMC
mkdir models
cd models
mkdir resnet101-Aic
```
Then put the pretrained model under this folder and run:
```
cd ELECTRICITY-MTMC
bash test.sh
```
The final results will locate at path ./exp/track3.txt

## Training
If you want to train the model by yourself, please first generate training sets through:
```
bash ./prepare.sh
```
Then run:
```
bash ./train.sh
```
You will get trained model under path ./models/resnet101-Aic
Finally run:
```
bash ./test.sh
```
The final results will locate at path ./exp/track3.txt
## Performance

The speed is tested on four 2080Ti GPUs.

* Speed: 0.345x real-time (it means we only need 0.345 second to tackle 1 second video)
* IDF1: 0.4616

## License

See `LICENSE`. Please read before use.

## Description
This is a PyTorch Re-Implementation of [EAST: An Efficient and Accurate Scene Text Detector](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_EAST_An_Efficient_CVPR_2017_paper.pdf).

* Only RBOX part is implemented.
* Using dice loss instead of class-balanced cross-entropy loss. Some codes refer to [argman/EAST](https://github.com/argman/EAST) and [songdejia/EAST](https://github.com/songdejia/EAST)
* The pre-trained model provided achieves __82.79__ F-score on ICDAR 2015 Challenge 4 using only the 1000 images. see [here](http://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_info&task=1&m=52405) for the detailed results.

| Model | Loss | Recall | Precision | F-score | 
| - | - | - | - | - |
| Original | CE | 72.75 | 80.46 | 76.41 |
| Re-Implement | Dice | 81.27 | 84.36 | 82.79 |

## Prerequisites
Only tested on
* Anaconda3
* Python 3.7.1
* PyTorch 1.0.1
* Shapely 1.6.4
* opencv-python 4.0.0.21
* lanms 1.0.2

When running the script, if some module is not installed you will see a notification and installation instructions. __if you failed to install lanms, please update gcc and binutils__. The update under conda environment is:

    conda install -c omgarcia gcc-6
    conda install -c conda-forge binutils

The original lanms code has a bug in ```normalize_poly``` that the ref vertices are not fixed when looping the p's ordering to calculate the minimum distance. We fixed this bug in [LANMS](https://github.com/SakuraRiven/LANMS) so that anyone could compile the correct lanms. However, this repo still uses the original lanms.

## Installation
### 1. Clone the repo

```
git clone https://github.com/SakuraRiven/EAST.git
cd EAST
```

### 2. Data & Pre-Trained Model
* Download Train and Test Data: [ICDAR 2015 Challenge 4](http://rrc.cvc.uab.es/?ch=4&com=downloads). Cut the data into four parts: train_img, train_gt, test_img, test_gt.

* Download pre-trained VGG16 from PyTorch: [VGG16](https://drive.google.com/open?id=1HgDuFGd2q77Z6DcUlDEfBZgxeJv4tald) and our trained EAST model: [EAST](https://drive.google.com/open?id=1AFABkJgr5VtxWnmBU3XcfLJvpZkC2TAg). Make a new folder ```pths``` and put the download pths into ```pths```
  
```
mkdir pths
mv east_vgg16.pth vgg16_bn-6c64b313.pth pths/
```

Here is an example:
```
.
├── EAST
│   ├── evaluate
│   └── pths
└── ICDAR_2015
    ├── test_gt
    ├── test_img
    ├── train_gt
    └── train_img
```
## Train
Modify the parameters in ```train.py``` and run:
```
CUDA_VISIBLE_DEVICES=0,1 python train.py
```
## Detect
Modify the parameters in ```detect.py``` and run:
```
CUDA_VISIBLE_DEVICES=0 python detect.py
```
## Evaluate
* The evaluation scripts are from [ICDAR Offline evaluation](http://rrc.cvc.uab.es/?ch=4&com=mymethods&task=1) and have been modified to run successfully with Python 3.7.1.
* Change the ```evaluate/gt.zip``` if you test on other datasets.
* Modify the parameters in ```eval.py``` and run:
```
CUDA_VISIBLE_DEVICES=0 python eval.py
```

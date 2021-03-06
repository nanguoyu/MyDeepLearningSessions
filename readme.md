
This is a project for re-implementing and re-training some famous deep vision models on 
tiny-Imagenet [[Introduction]](http://cs231n.stanford.edu/reports/2017/pdfs/930.pdf) 
[[Download Dataset]](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
with AWS EC2 instances.

Models are implemented by Pytorch. The general training scripts are modified from [
pytorch/examples](https://github.com/pytorch/examples), supporting 
distributed training in single-machine multi-GPU and multi-machine.

I only have a GTX 1650 laptop which is hard to train deep models on big dataset within an accepted time. 
As a result, I focus on AWS EC2 P3 instances for a complete training with a pre-configured 
Deep Learning AMI (Ubuntu 18.04) Version 56.0. 

That's saying, you can train models with the same/similar GPU-supported EC2 instances and AMI.

## Create the basic environment

### Use an AWS EC2 Instance

1. Create a `p3.2xlarge` instance with `Deep Learning AMI (Ubuntu 18.04) Version 56.0 ami-0e14490e647237ed6`

2. Attach an enough EBS disk >= `200GB`

3. SSH to the machine

### Or use your local machine 

TODO

## Setup

Run this script, the required libraries would be installed. Currently, this script and 
even the whole project are tightly dependent on a specific AMI. We will introduce docker supports in the future.

```Shell
source script/setup.sh
```

## Train models


### Prepare tinyImagenet

Download dataset
```Shell
bash script/tinyImagenetDataset.sh
```

### Train Alexnet on the tiny-Imagenet

```Shell
python3 train_image_classification.py  -a alexnet --lr 0.01 -b 256 --num-classes 200 --multiprocessing-distributed --world-size 1 --dist-url tcp://127.0.0.1:10086 --rank 0  ./tiny-imagenet-200
```

### Train VGG16 on the tiny-Imagenet

Supports: vgg11, vgg13, vgg16, and vgg19


```Shell
python3 train_image_classification.py  -a vgg16 --lr 0.01 -b 64 --num-classes 200 --multiprocessing-distributed --world-size 1 --dist-url tcp://127.0.0.1:10086 --rank 0  ./tiny-imagenet-200
```

### Train GoogLeNet on the tiny-Imagenet


```Shell
python3 train_image_classification.py  -a googlenet --lr 0.01 -b 128 --num-classes 200 --multiprocessing-distributed --world-size 1 --dist-url tcp://127.0.0.1:10086 --rank 0  ./tiny-imagenet-200
```


### Train Resnet on the tiny-Imagenet

Supports: resnet18, resnet34, resnet50, resnet101, resnet152


```Shell
python3 train_image_classification.py  -a resnet18 --lr 0.1 -b 256 --num-classes 200 --multiprocessing-distributed --world-size 1 --dist-url tcp://127.0.0.1:10086 --rank 0  ./tiny-imagenet-200
```

### Train Mobilenet on the tiny-Imagenet

Supports: mobilenet_v2


```Shell
python3 train_image_classification.py  -a mobilenet_v2 --lr 0.045 --wd 4e-5 -b 256 --num-classes 200 --multiprocessing-distributed --world-size 1 --dist-url tcp://127.0.0.1:10086 --rank 0  ./tiny-imagenet-200
```

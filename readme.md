

## Setup environment


```Shell
source script/setup.sh
```

## Train models


### tinyImagenet

Download dataset
```Shell
bash script/tinyImagenetDataset.sh
```

### Train Alexnet on the tiny-Imagenet

```Shell
python3 train_image_classification.py  -a alexnet --lr 0.01 -b 256 --multiprocessing-distributed --world-size 1 --dist-url tcp://127.0.0.1:10086 --rank 0  ./tiny-imagenet-200
```

### Train VGG15 on the tiny-Imagenet


```Shell
python3 train_image_classification.py  -a vgg16 --lr 0.01 -b 64 --multiprocessing-distributed --world-size 1 --dist-url tcp://127.0.0.1:10086 --rank 0  ./tiny-imagenet-200
```
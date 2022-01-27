# /bin/sh

sudo apt update -y && sudo apt install unzip wget python3 python3-pip libjpeg-dev zlib1g-dev -y \
&& wget http://cs231n.stanford.edu/tiny-imagenet-200.zip && unzip tiny-imagenet-200.zip \
&& python3 reformatValdata.py


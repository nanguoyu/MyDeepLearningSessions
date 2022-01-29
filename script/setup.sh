# /bin/sh

sudo add-apt-repository ppa:ubuntu-toolchain-r/test \
&& sudo apt update -y && sudo apt install unzip wget libjpeg-dev zlib1g-dev gcc-9 -y \
&& sudo apt-get install --only-upgrade libstdc++6 \
&& source activate pytorch_latest_p37 \
&& pip3 install -r requirements.txt

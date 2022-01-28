# /bin/sh

sudo apt update -y && sudo apt install unzip wget libjpeg-dev zlib1g-dev -y && source activate pytorch_latest_p37 \
&& pip3 install -r requirements.txt

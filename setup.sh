#! /usr/bin/bash

apt update

apt install zip unzip

python3 -m pip install cityscapesscripts

pip install shapely

python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

pip install tensorboard

pip install opencv-python

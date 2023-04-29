#! /usr/bin/bash
sudo apt install python3-pip

pip install torch==1.13

pip install torchvision

python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

python3 -m pip install cityscapesscripts

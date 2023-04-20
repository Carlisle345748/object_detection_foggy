#! /usr/bin/bash

apt-get update
apt-get install gcc, g++
apt-get install nvidia-cuda-dev

pip install shapely
pip install torch, torchvision
pip install opencv-python
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'


# Download and prepare dataset
function download_and_unzip() {
    datasets_name="$1"
    zip_file="$2"
    python scripts/download.py -d 'datasets' "$zip_file"
    echo "unzipping $zip_file..."
    unzip -q "datasets/$zip_file" -d "datasets/$datasets_name"
    rm "datasets/$datasets_name/license.txt" "datasets/$datasets_name/README" "datasets/$zip_file"
    echo "unzipping $zip_file finished"
}

if [ ! -d "datasets/cityscapes/gtFine" ]; then
    download_and_unzip cityscapes gtFine_trainvaltest.zip
fi

if [ ! -d "datasets/cityscapes/leftImg8bit" ]; then
    download_and_unzip cityscapes leftImg8bit_trainvaltest.zip
fi

if [ ! -d "datasets/cityscapes_foggy/leftImg8bit" ]; then
    download_and_unzip cityscapes_foggy leftImg8bit_trainval_foggyDBF.zip
    mv datasets/cityscapes_foggy/leftImg8bit_foggyDBF datasets/cityscapes_foggy/leftImg8bit
    rm datasets/cityscapes_foggy/foggy_trainval_refined_filenames.txt
    rm datasets/cityscapes_foggy/README_foggyDBF.md
fi

if [ ! -d "datasets/cityscapes_foggy/gtFine" ]; then
    cp -r datasets/cityscapes/gtFine datasets/cityscapes_foggy/gtFine
fi





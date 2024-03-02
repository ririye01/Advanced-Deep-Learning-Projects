#!/bin/sh

#SBATCH -J imagenet-vgg          # Job Name
#SBATCH -p batch                 # (debug or batch)
#SBATCH --exclusive
#SBATCH -o runjob.out            # Output file name
#SBATCH --gres=gpu:8             # Number of GPUs to use
#SBATCH --mail-user ririye@smu.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=200GB

echo "Parsing ImageNet Data"
#######################################################################################################################
# Inspiration from: https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh
# # script to extract ImageNet dataset
# ILSVRC2012_img_train.tar (about 138 GB)
# ILSVRC2012_img_val.tar (about 6.3 GB)
# make sure ILSVRC2012_img_train.tar & ILSVRC2012_img_val.tar in your current directory
#
#  Adapted from:
#  https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md
#  https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4
#
#  imagenet/train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#  imagenet/val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
#
# Navigate to current working directory in case we are not there
cd ~/work/Advanced-Deep-Learning-Projects/lab2_transformers_and_transfer_learning
#
# Make ImageNet directory to keep data stored there
#
mkdir imagenet
#
# Extract the training data:
#
# Create train directory; move .tar file; change directory
mkdir imagenet/train && mv ILSVRC2012_img_train.tar imagenet/train/ && cd imagenet/train
# Extract training set; remove compressed file
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
#
# At this stage imagenet/train will contain 1000 compressed .tar files, one for each category
#
# For each .tar file:
#   1. create directory with same name as .tar file
#   2. extract and copy contents of .tar file into directory
#   3. remove .tar file
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
#
# This results in a training directory like so:
#
#  imagenet/train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#
# Change back to original directory
cd ~/work/Advanced-Deep-Learning-Projects/lab2_transformers_and_transfer_learning
#
# Extract the validation data and move images to subfolders:
#
# Create validation directory; move .tar file; change directory; extract validation .tar; remove compressed file
mkdir imagenet/val && mv ILSVRC2012_img_val.tar imagenet/val/ && cd imagenet/val && tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
# get script from soumith and run; this script creates all class directories and moves images into corresponding directories
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
#
# This results in a validation directory like so:
#
#  imagenet/val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
#
#
# Check total files after extract
#
#  $ find train/ -name "*.JPEG" | wc -l
#  1281167
#  $ find val/ -name "*.JPEG" | wc -l
#  50000
#######################################################################################################################



# Move to Lab2 Transformers & Transfer Learning Directory
cd /users/ririye/work/Advanced-Deep-Learning-Projects/lab2_transformers_and_transfer_learning

#######################################################################################################################
# Run script for training PyTorch model
echo "Running torchrun"
torchrun --nproc_per_node 8 --master-port 29503 train_ddp.py
#######################################################################################################################

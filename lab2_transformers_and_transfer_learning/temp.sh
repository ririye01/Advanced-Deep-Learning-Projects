#!/bin/sh

#SBATCH -J imagenet-parse        # Job Name
#SBATCH -p batch                 # (debug or batch)
#SBATCH --exclusive
#SBATCH -o downloadjob.out       # Output file name
#SBATCH --gres=gpu:1             # Number of GPUs to use
#SBATCH --mail-user ririye@smu.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=300GB

# ImageNet Directory
cd imagenet/train
if [ ! -f ./ILSVRC2012_img_train.tar ] || [ $(stat -c%s "./ILSVRC2012_img_train.tar") -lt 107374182400 ]; then
    curl -O https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
fi

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


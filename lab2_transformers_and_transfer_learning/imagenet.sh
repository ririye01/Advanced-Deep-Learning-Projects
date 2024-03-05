#!/bin/sh
#SBATCH -J imagenet-download        # Job Name
#SBATCH -p batch                 # (debug or batch)
#SBATCH --exclusive
#SBATCH -o downloadjob-tar.out       # Output file name
#SBATCH --gres=gpu:1             # Number of GPUs to use
#SBATCH --mail-user ririye@smu.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=300GB
#SBATCH --error=downloadjob-tar.err      # file to collect standard errors


echo "Downloading ImageNet"
#######################################################################################################################
# Download 2012 ImageNet dataset from source
curl -O https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
curl -O https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

# Download Dev kit for task 1 & 2
curl -O https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
#######################################################################################################################
echo "ImageNet Download Complete."




#!/bin/sh

#SBATCH -J imagenet-vgg          # Job Name
#SBATCH -p batch                 # (debug or batch)
#SBATCH --exclusive
#SBATCH -o runjob.out            # Output file name
#SBATCH --gres=gpu:8             # Number of GPUs to use
#SBATCH --mail-user ririye@smu.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=200GB


# Move to Lab2 Transformers & Transfer Learning Directory
cd ~/work/Advanced-Deep-Learning-Projects/lab2_transformers_and_transfer_learning

#######################################################################################################################
# Run script for training PyTorch model
echo "Running torchrun"
torchrun --nproc_per_node 8 --master-port 29503 train_ddp.py
#######################################################################################################################

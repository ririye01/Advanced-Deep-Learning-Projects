#!/bin/sh

#SBATCH -J mnist-run             # Job Name
#SBATCH -p batch                 # (debug or batch)
#SBATCH -o runjob.out            # Output file name
#SBATCH --gres=gpu:4             # Number of GPUs to use
#SBATCH --mail-user ririye@smu.edu
#SBATCH --mail-type=ALL
#SBATCH --exclusive
#SBATCH --mem=100GB


# Move to Lab2 Transformers & Transfer Learning Directory
cd ~/work/Advanced-Deep-Learning-Projects/lab2_transformers_and_transfer_learning

# Activate conda environment
# module load conda
# conda create -n cs8321-lab2
# conda activate cs8321-lab2
# pip install -r requirements.txt
module load conda
conda activate cs8321-lab1

#######################################################################################################################
# Run script for training PyTorch model
echo "Running torchrun"
torchrun --nproc_per_node 8 --master-port 29503 train_ddp.py
#######################################################################################################################

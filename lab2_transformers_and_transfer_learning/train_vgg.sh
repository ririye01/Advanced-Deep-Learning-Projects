#!/bin/sh

#SBATCH -J ¯\_(ツ)_/¯            # Job Name
#SBATCH -p batch                 # (debug or batch)
#SBATCH --exclusive
#SBATCH -o runjob.out            # Output file name
#SBATCH --gres=gpu:8             # Number of GPUs to use
#SBATCH --mail-user ririye@smu.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=200GB

cd /users/ririye/work/Advanced-Deep-Learning-Projects/lab2_transformers_and_transfer_learning
torchrun --nproc_per_node 8 --master-port 29503 train_ddp.py

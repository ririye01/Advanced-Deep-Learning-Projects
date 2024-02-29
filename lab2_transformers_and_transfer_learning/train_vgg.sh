#!/bin/bash
#SBATCH --job-name=vgg16_imagenet
#SBATCH --output=vgg16_train_%j.out
#SBATCH --error=vgg16_train_%j.err
#SBATCH -N 4
#SBATCH -c 128
#SBATCH --pty
#SBATCH --ntasks-per-node=4  # 4 tasks per node, one for each GPU
#SBATCH --cpus-per-task=32  # Adjusted for 4 tasks per node
#SBATCH --mem=256G
#SBATCH --mail-user=ririye@mail.smu.edu
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=2-00:00:00        # total run time limit (D-HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails

srun --ntasks=16 python -m torch.distributed.launch --nproc_per_node=4 train_ddp.py

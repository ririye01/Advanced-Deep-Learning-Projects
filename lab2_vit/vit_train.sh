#!/bin/sh

#SBATCH -J ILYHOWL!
#SBATCH -p batch
#SBATCH --exclusive
#SBATCH -o runjob.out
#SBATCH --gres=gpu:1
#SBATCH --mem=200G

cd /users/tdohm/work/Advanced-Deep-Learning-Projects/lab2_vit
source /users/tdohm/venvs/pytorch/bin/activate
python vit_model.py